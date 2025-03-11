import time
import pandas as pd
import difflib
import networkx as nx
import math
import numpy as np
from itertools import combinations
import leidenalg as la
import igraph as ig
import multiprocessing
from itertools import islice
import ast
import pickle

def SetConv(txt):
    # convert array of strings into sets for the route ids' stored in the .csv
    return set() if txt == 'set()' else ast.literal_eval(txt)

def travel_time_estimator(lat1, lon1, lat2, lon2):
    sin, cos, sqrt, atan2, radians = math.sin, math.cos, math.sqrt, math.atan2, math.radians
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c  # km
    speed = 10  # km/h

    return (distance / speed) * 60

def init_graph_builder(stop_ls_df):
    gb = stop_ls_df.groupby('line_name')
    G = nx.DiGraph()
    for each_group in gb.groups:
        each_group = gb.get_group(each_group)
        each_group = each_group.sort_values(by='sequence', ascending=True)
        row_iterator = each_group.iterrows()
        _, last_row = next(
            row_iterator)  # take the first row, and the pointer of row_iterator would be at the second row.
        curr_route_id = last_row['line_name']  # stop name of the first row
        G.add_node(last_row['station_name'], long=last_row['coord_x'], lat=last_row['coord_y'], route={curr_route_id})
        for _, row in row_iterator:
            curr_stop_name = row['station_name']
            if not G.has_node(curr_stop_name):
                # add the node into the graph if the node is not in the graph, and add three node attributes
                G.add_node(curr_stop_name, long=row['coord_x'], lat=row['coord_y'], route={curr_route_id})
            else:
                # if the node is already included in the graph, only append its route id to the existed node in the current graph.
                G.nodes[curr_stop_name]['route'].add(curr_route_id)

            # travel time between two consecutive rows in avl data
            travel_time = travel_time_estimator(last_row['coord_y'], last_row['coord_x'], row['coord_y'],
                                                row['coord_x'])
            if last_row['station_name'] != curr_stop_name:  # if these two consecutive rows are different bus stops
                if not G.has_edge(last_row['station_name'],
                                  curr_stop_name):  # if the edge from last row to current row not in the current graph, add it.
                    G.add_edge(last_row['station_name'], curr_stop_name, travel_time=travel_time, route={curr_route_id})
                else:
                    G.edges[last_row['station_name'], curr_stop_name]['route'].add(
                        curr_route_id)  # append route_id to the route_id set of the existing edge
            last_row = row  # assign current row to last row and continue the iteration
    return G

def string_similar(s1, s2):
    return difflib.SequenceMatcher(lambda x: x == " ", s1, s2).quick_ratio()


def rename_stop(each_stop_ic, station_ls_gd):
    # compute the similarity between row['stop_name'] and each of station name in the station list fetched from GaoDe
    ind = station_ls_gd.station_name.apply(lambda x: string_similar(each_stop_ic, x)).idxmax()
    return station_ls_gd.loc[ind, 'station_name']


def to_undirected_sum(G, weight='flow'):
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


def graph_clustering(G):
    coms_leiden_igraph = la.find_partition(ig.Graph.from_networkx(G),
                                           partition_type=la.ModularityVertexPartition,
                                           weights='flow', n_iterations=10, max_comm_size=60)
    print(coms_leiden_igraph.modularity)
    coms_leiden_df = pd.DataFrame()
    coms_leiden_df['station_name'] = np.array(G.nodes())
    coms_leiden_df['membership'] = coms_leiden_igraph.membership

    coms_leiden = coms_leiden_df.groupby('membership')['station_name'].apply(list).reset_index(name='community')
    coms_leiden['stop_num'] = coms_leiden.apply(lambda x: len(x['community']), axis=1)
    return coms_leiden


def unaddressed_flow(coms_ls, df):
    coms_df = [df[df.o_station_name.isin(each_com) & df.d_station_name.isin(each_com)] for each_com in
               coms_ls.community]
    od_flow_index = np.concatenate([each_com_df.index.values for each_com_df in coms_df], axis=0)

    return df[~df.index.isin(od_flow_index)], coms_df


def filter_not_in_G(com_stop_ls, g_stop_ls):
    """
    com_stop_ls is the list of bus stops in the identified community from the flow network (smart card data).
    g_stop_ls is the list of bus stops from the graph constructed from avl data.
    """
    x = np.array(com_stop_ls)
    y = np.array(g_stop_ls)

    res = x[np.isin(x, y)]

    return list(res)


def extended_subgraph(params):
    G_df, nodelist = params[0], params[1]
    G = nx.from_pandas_edgelist(G_df, 'source', 'target', ['travel_time', 'route'], create_using=nx.DiGraph())
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


def k_shortest_paths(G, source, target, k, weight='travel_time'):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

if __name__ == '__main__':

    stop_ls_gd = pd.read_csv('./data/nanjing_station_GD.csv')

    G = init_graph_builder(stop_ls_gd)

    df_od = pd.read_csv('./data/OD_flow_duration_f.csv')
    terminal_weight_df = pd.read_csv('./data/terminal_weight_df.csv')
    stop_weight_df_rename = pd.read_csv('./data/stop_weight_df_rename.csv')
    stop_weight_df_merged = pd.read_csv('./data/stop_weight_df_merged.csv')
    stop_weight_df_merged.route = stop_weight_df_merged.route.apply(SetConv)

    G_df = pd.read_csv('./data/gaode_top_net.csv')
    G_df.route = G_df.route.apply(SetConv)

    print('start params preparation')
    start_params = time.time()

    total_flow = df_od.flow.sum()

    coms_ls = []
    df_exclude_ls = []

    while (df_od.flow.sum() / total_flow) > 0.001:
        H = nx.from_pandas_edgelist(df_od, 'o_station_name', 'd_station_name', edge_attr=['flow'],
                                    create_using=nx.DiGraph())

        H_undirected = to_undirected_sum(H)

        coms_leiden = graph_clustering(H_undirected)

        coms_leiden = coms_leiden[coms_leiden.community.map(len) >= 20]

        coms_ls.append(coms_leiden)

        df_od, df_od_exclude = unaddressed_flow(coms_leiden, df_od)

        df_exclude_ls.extend(df_od_exclude)

    coms_leiden = pd.concat(coms_ls, ignore_index=True)
    coms_leiden = coms_leiden.sort_values(by='community', key=lambda x: x.str.len(), ascending=False)

    num_cores = multiprocessing.cpu_count() - 2

    params_sg = [[G_df, each_com] for each_com in coms_leiden.community]
    with multiprocessing.Pool(num_cores) as pool:
        sg_df_ls = pool.map(extended_subgraph, params_sg)
        pool.close()
        pool.join()

    params = []
    print('Number of clusters:{}'.format(len(coms_leiden)))
    total_route_num = 0
    max_route_limit = 810  # single-direction, current route set of Nanjing is 405 bi-direction routes
    for i in range(len(coms_leiden)):
        com_id = i
        curr_com_ls = coms_leiden.community[i]
        df_com_exclude = df_exclude_ls[i]

        df_com_od = df_com_exclude[
            df_com_exclude.o_station_name.isin(curr_com_ls) & df_com_exclude.d_station_name.isin(curr_com_ls)]

        sg_df = sg_df_ls[i]
        sg = nx.from_pandas_edgelist(sg_df, 'source', 'target', ['travel_time', 'route'], create_using=nx.DiGraph())
        ext_com_ls = list(sg.nodes())

        com_terminal_weight_df = terminal_weight_df[terminal_weight_df.station_name.isin(ext_com_ls)]
        com_stop_weight_df = stop_weight_df_rename[stop_weight_df_rename.station_name.isin(ext_com_ls)]
        com_stop_merged_df = stop_weight_df_merged[stop_weight_df_merged.station_name.isin(ext_com_ls)]

        route_num = int(np.round((df_com_od.flow.sum() / total_flow) * max_route_limit))
        if (route_num < 1) and total_route_num >= max_route_limit + 10:
            continue
        elif route_num < 1:
            route_num = 1

        total_route_num += route_num

        if len(com_terminal_weight_df) >= 8:
            # params.append(
            #     [com_id, route_num, sg_df, com_stop_merged_df, com_terminal_weight_df, 'travel_time', df_com_od])
            params.append(
                [com_id, route_num, sg_df, com_stop_merged_df, com_stop_merged_df, 'travel_time', df_com_od])
        else:
            params.append(
                [com_id, route_num, sg_df, com_stop_merged_df, com_stop_merged_df, 'travel_time', df_com_od])

    print('Total route number for the parameter: {}'.format(total_route_num))

    with open('./tmp_results/tmp_parameters.pkl', 'wb') as f:
        pickle.dump(params, f)