"""
assign number of routes according to passenger demand or number of nodes
"""
import time
import pandas as pd
import networkx as nx
import numpy as np
from itertools import combinations
import leidenalg as la
import igraph as ig
import multiprocessing
import pickle
import math
import matplotlib.pyplot as plt
import matplotlib
import random
import geopandas as gpd
from datashader.bundling import hammer_bundle  # key for bundling


def to_undirected_sum(G, clustering_type):
    """
    convert directed graph to the undirected and sum the attribute,
    nx.to_undirected function only use the edge attribute with the max value
    """
    weight = clustering_type
    UG = G.to_undirected()
    for node in G:
        for ngbr in nx.neighbors(G, node):
            if node in nx.neighbors(G, ngbr):
                UG.edges[node, ngbr][weight] = (
                        G.edges[node, ngbr][weight] + G.edges[ngbr, node][weight]
                )
    return UG


def to_undirected_sum_viz(G, weight='flow'):
    """
    convert directed graph to the undirected and sum the attribute,
    nx.to_undirected function only use the edge attribute with the max value
    """
    UG = G.to_undirected()
    for node in G:
        for ngbr in nx.neighbors(G, node):
            if node in nx.neighbors(G, ngbr):
                UG.edges[node, ngbr][weight] = (
                        G.edges[node, ngbr][weight] + G.edges[ngbr, node][weight]
                )
    return UG


def graph_clustering(G, level, clustering_type):
    coms_leiden_igraph = la.find_partition(ig.Graph.from_networkx(G),
                                           partition_type=la.ModularityVertexPartition,
                                           weights=clustering_type, n_iterations=10,
                                           max_comm_size=60)
    each_layer_modularity = coms_leiden_igraph.modularity
    coms_leiden_df = pd.DataFrame()
    coms_leiden_df['station_name'] = np.array(G.nodes())
    coms_leiden_df['membership'] = coms_leiden_igraph.membership

    coms_leiden = coms_leiden_df.groupby('membership')['station_name'].apply(
        list).reset_index(name='community')
    coms_leiden['stop_num'] = coms_leiden.apply(lambda x: len(x['community']), axis=1)
    coms_leiden['level'] = level
    return coms_leiden, each_layer_modularity


def extended_subgraph(params):
    G_df, nodelist = params[0], params[1]
    G = nx.from_pandas_edgelist(G_df, 'source', 'target', ['travel_time'],
                                create_using=nx.DiGraph())
    H = nx.to_undirected(G)

    paths = set()
    for nodes in combinations(nodelist, r=2):
        if nx.has_path(H, *nodes):
            # paths = paths.union(set(nx.shortest_path(H, *nodes, weight='travel_time')))
            paths = paths.union(set(nx.shortest_path(H, *nodes)))
        else:
            paths = paths.union(nodes)
    sG = nx.subgraph(G, paths)

    return nx.to_pandas_edgelist(sG)


def filter_not_in_G(com_stop_ls, g_stop_ls):
    """
    com_stop_ls is the list of bus stops in the identified community from the flow network (smart card data).
    g_stop_ls is the list of bus stops from the graph constructed from avl data.
    """
    x = np.array(com_stop_ls)
    y = np.array(g_stop_ls)

    res = x[np.isin(x, y)]

    return list(res)


def attraction(params):
    df = params[0]
    df_links = params[1]
    G = nx.from_pandas_edgelist(df_links, source='from', target='to',
                                edge_attr=['travel_time'],
                                create_using=nx.DiGraph())
    for each_index, each_row in df.iterrows():
        df.at[each_index, 're_flow'] = each_row['flow'] / nx.shortest_path_length(G,
                                                                                  source=
                                                                                  each_row[
                                                                                      'o_station_name'],
                                                                                  target=
                                                                                  each_row[
                                                                                      'd_station_name'],
                                                                                  weight='travel_time')
    return df


def multi_leiden(df_od, clustering_type):
    total_flow = df_od.flow.sum()
    coms_ls = []
    df_exclude_ls = []
    # passenger flow network at each level of clustering
    df_each_level_ls = []

    level = 0

    average_weighted_modularity = 0

    while (df_od.flow.sum() / total_flow) > 0.001:
        df_each_level_ls.append(df_od)
        H = nx.from_pandas_edgelist(df_od, 'o_station_name', 'd_station_name',
                                    edge_attr=[clustering_type],
                                    create_using=nx.DiGraph())

        H_undirected = to_undirected_sum(H, clustering_type)

        coms_leiden, each_layer_modularity = graph_clustering(H_undirected, level, clustering_type)

        coms_leiden = coms_leiden[coms_leiden.community.map(len) >= 12]
        coms_df = np.array([df_od[df_od.o_station_name.isin(each_com) & df_od.d_station_name.isin(each_com)].flow.sum()
                   for
                   each_com in
                   coms_leiden.community]) / total_flow
        # coms_leiden = coms_leiden[coms_df < 0.06]


        coms_ls.append(coms_leiden)

        df_od, df_od_exclude = unaddressed_flow(coms_leiden, df_od)

        df_exclude_ls.extend(df_od_exclude)

        level += 1
        each_layer_weight = [each_exclude_df.flow.sum() for each_exclude_df in
                             df_od_exclude]
        each_layer_weight = sum(each_layer_weight)

        average_weighted_modularity += (
                                                   each_layer_weight / total_flow) * each_layer_modularity

    print(average_weighted_modularity)
    coms_leiden = pd.concat(coms_ls, ignore_index=True)
    # coms_leiden = coms_leiden.sort_values(by='community', key=lambda x: x.str.len(),
    #                                       ascending=False)

    return coms_leiden, df_exclude_ls, df_each_level_ls, average_weighted_modularity


def unaddressed_flow(coms_ls, df):
    coms_df = [df[df.o_station_name.isin(each_com) & df.d_station_name.isin(each_com)] for
               each_com in
               coms_ls.community]
    if len(coms_df) > 0:
        od_flow_index = np.concatenate(
            [each_com_df.index.values for each_com_df in coms_df], axis=0)
    else:
        return pd.DataFrame().reindex_like(df), coms_df

    return df[~df.index.isin(od_flow_index)], coms_df


if __name__ == '__main__':

    # build road network
    df_links = pd.read_csv('./data/mumford3_links.txt')
    G = nx.from_pandas_edgelist(df_links, source='from', target='to',
                                edge_attr=['travel_time'],
                                create_using=nx.DiGraph())

    G_df = df_links.rename(columns={'from': 'source', 'to': 'target'})

    df_nodes = pd.read_csv('./data/mumford3_nodes.txt')
    df_od = pd.read_csv('./data/mumford3_demand.txt')

    df_od = pd.merge(df_od, df_nodes[['id', 'lat', 'lon']], how='inner', left_on='from',
                     right_on='id')
    df_od = pd.merge(df_od, df_nodes[['id', 'lat', 'lon']], how='inner', left_on='to',
                     right_on='id')

    df_od = df_od[['from', 'to', 'demand', 'lat_x', 'lon_x', 'lat_y', 'lon_y']].rename(
        columns={'from': 'o_station_name', 'to': 'd_station_name', 'demand': 'flow',
                 'lat_x': 'o_lat',
                 'lon_x': 'o_long',
                 'lat_y': 'd_lat', 'lon_y': 'd_long'})

    df_od['re_flow'] = 0

    cores = multiprocessing.cpu_count() - 1
    partitions = cores
    df_od_split = np.array_split(df_od, partitions)
    params = []
    for i in range(partitions):
        params.append([df_od_split[i], df_links])
    with multiprocessing.Pool(cores) as pool:
        # fitness_ls, df_od_ls = zip(*pool.map(fitness, params))
        df_od_ls = pool.map(attraction, params)
        pool.close()
        pool.join()
    df_od = pd.concat(df_od_ls, ignore_index=True)

    stop_weight_df = df_od.groupby(by='o_station_name').agg(
        {'flow': 'sum', 'o_lat': 'first', 'o_long': 'first'}).reset_index()
    stop_weight_df = stop_weight_df.rename(
        columns={'o_station_name': 'station_name', 'flow': 'total_flow', 'o_lat': 'lat',
                 'o_long': 'long'})

    print('start params preparation')
    start_params = time.time()

    total_flow = df_od.flow.sum()

    # graph_clutering flow or re_flow (attraction)
    clustering_type = 're_flow'
    # experiment multiple times of graph clustering to reduce the effect of randomness in local node movements
    clustering_times = 0
    coms_leiden, df_exclude_ls, df_each_level_ls, average_weighted_modularity = None, None, None, 0
    while clustering_times < 30:
        print('current_graph_clustering_iteration:{}'.format(clustering_times))
        tmp_coms_leiden, tmp_df_exclude_ls, tmp_df_each_level_ls, tmp_average_modularity = multi_leiden(
            df_od,clustering_type)

        if tmp_average_modularity > average_weighted_modularity:
            coms_leiden, df_exclude_ls, df_each_level_ls, average_weighted_modularity = tmp_coms_leiden, tmp_df_exclude_ls, tmp_df_each_level_ls, tmp_average_modularity

        clustering_times += 1

    num_cores = multiprocessing.cpu_count() - 2

    inaccessibility_penalty = 24.74
    # params_sg = [[G_df, each_com] for each_com in coms_leiden.community]
    # with multiprocessing.Pool(num_cores) as pool:
    #     sg_df_ls = pool.map(extended_subgraph, params_sg)
    #     pool.close()
    #     pool.join()

    sg_df_ls = [G_df] * len(coms_leiden)

    params = []
    print('Number of clusters:{}'.format(len(coms_leiden)))
    total_route_num = 0
    max_route_limit = 60  # single-direction, current route set of Nanjing is 405 bi-direction routes

    # s = [each_com_df.flow.sum() for each_com_df in df_exclude_ls]
    # s_order = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
    com_count = 0
    # total number of nodes in clusters, not the number of nodes of the complete network
    num_nodes_c = [len(each_com) for each_com in coms_leiden.community]
    # s_order = sorted(range(len(num_nodes_c)), key=lambda k: num_nodes_c[k], reverse=True)
    num_nodes_c = sum(num_nodes_c)

    # assign num of routes according to num of nodes?
    assign_by_nodes = False
    for i in range(len(coms_leiden)):
    # for i in s_order:

        curr_com_ls = coms_leiden.community[i]
        curr_level = coms_leiden.level[i]
        curr_membership = coms_leiden.membership[i]
        df_com_exclude = df_exclude_ls[i]
        com_id = str(curr_level) + '_' + str(curr_membership)
        # df_com_od = df_com_exclude[
        #     df_com_exclude.o_station_name.isin(
        #         curr_com_ls) & df_com_exclude.d_station_name.isin(curr_com_ls)]

        sg_df = sg_df_ls[i]
        sg = nx.from_pandas_edgelist(sg_df, 'source', 'target', ['travel_time'],
                                     create_using=nx.DiGraph())
        ext_com_ls = list(sg.nodes())

        com_stop_merged_df = stop_weight_df[stop_weight_df.station_name.isin(ext_com_ls)]
        com_df_terminal = stop_weight_df[stop_weight_df.station_name.isin(curr_com_ls)]
        com_stop_merged_df['com_flow'] = 0
        com_stop_merged_df.loc[com_stop_merged_df.station_name.isin(curr_com_ls),'com_flow'] = com_stop_merged_df[
            com_stop_merged_df.station_name.isin(curr_com_ls)].total_flow

        if assign_by_nodes:
            route_num = math.floor((len(curr_com_ls) / num_nodes_c) * max_route_limit)

            # route_num = int(np.round((len(curr_com_ls) / num_nodes_c) * max_route_limit))
        else:
            route_num = int(
                math.floor((df_com_exclude.flow.sum() / total_flow) * max_route_limit))

        if total_route_num >= max_route_limit:
            break

        if (route_num < 1):
            route_num = 1

        total_route_num += route_num

        params.append(
            [com_id, route_num, sg_df, com_stop_merged_df, com_df_terminal,
             'travel_time', df_com_exclude,curr_com_ls])
        com_count += 1

    i = 0
    total_exclude_flow = sum([each_od_df.flow.sum() for each_od_df in df_exclude_ls])
    remain_route_num = max_route_limit - total_route_num
    while total_route_num < max_route_limit:
        if assign_by_nodes:
            add_route_num = int(round((len(params[i][7]) / num_nodes_c) * remain_route_num))
        else:
            add_route_num = int(round(
                (params[i][6].flow.sum() / total_exclude_flow) * remain_route_num))
        # if add_route_num < 1:
        #     add_route_num = 1
        params[i][1] = params[i][1] + add_route_num
        total_route_num += add_route_num
        i += 1
        if i >= len(params):
            i = 0


    for each in params:
        print('cluster {} has {} routes'.format(each[0], each[1]))

    print('Total route number for the parameter: {}'.format(total_route_num))

    with open('./tmp_results/tmp_parameters_mumford3_GCH.pkl', 'wb') as f:
        pickle.dump(params, f)

    with open('./tmp_results/tmp_parameters_mumford3_GCH_viz.pkl', 'wb') as f:
        pickle.dump(
            [coms_leiden, df_exclude_ls, df_each_level_ls, average_weighted_modularity],
            f)
