"""
The graph representing connectivity topology between bus stops are constructed using data scrapping from
GaoDe API.
"""
import time
import pandas as pd
import difflib
import dask.dataframe as dd
import networkx as nx
# from math import sin, cos, sqrt, atan2, radians
import math
import gc
# from cdlib import algorithms, viz
import numpy as np
from itertools import combinations
import leidenalg as la
import igraph as ig
import random
import copy
import multiprocessing
from networkx.classes.function import path_weight
from itertools import islice
import ctypes
import ast
import pickle
import sys


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


def unaddressed_flow(coms_ls, df):
    coms_df = [df[df.o_station_name.isin(each_com) & df.d_station_name.isin(each_com)] for
               each_com in
               coms_ls.community]
    od_flow_index = np.concatenate([each_com_df.index.values for each_com_df in coms_df],
                                   axis=0)

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


def k_shortest_paths(G, source, target, k, weight='travel_time'):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def normalize_weight(df, source_name, G):
    """
    df should be the dataframe of out-neighbors of the source node.
    """
    for index, each_row in df.iterrows():
        tmp_each_time = G.edges[source_name, each_row['station_name']]['travel_time']
        # each_route_ids = G.edges[source_name, each_row['station_name']]['route']
        if tmp_each_time >= 1:
            norm_weight = df.loc[index, 'total_flow'] / tmp_each_time
            if norm_weight > 1:
                df.at[index, 'total_flow'] = norm_weight
            else:
                df.at[index, 'total_flow'] = 1

        # df.at[index, 'route'] = each_route_ids


def one_route(df_terminal, weight_df, G, com_id, route_id):
    """
    df_terminal contains the dataframe with in-out-flow weights and coordinates for terminal stops.
    weight_df is the dataframe for all bus stops with weights and coordinates.
    G is graph truncated by the above alog.
    """
    route = []  # store
    longest_time = 0  # from the starting terminal to the initial terminal
    max_stop_num = 25
    min_stop_num = 12
    # max_travel_time = 60  # unit: minute
    # min_travel_time = 20

    if (len(df_terminal) == 0):
        return []
    stop_row = df_terminal.sample(n=1,
                                  weights='total_flow').copy()  # stop_row is a dataframe
    stop_row.loc[:, 'travel_time'] = 0
    route.append(stop_row)

    # drop the terminal from weight_df
    weight_df.drop(stop_row.index, inplace=True)

    stop_name = stop_row.station_name.values[0]

    out_nbrs = list(G.neighbors(stop_name))
    while (len(route) < max_stop_num) and (len(out_nbrs) > 0):
        # get the df of neighbors
        nbr_weight_df = weight_df[weight_df['station_name'].isin(out_nbrs)].copy()
        if len(nbr_weight_df) == 0:
            break

        # averaging the total flow weight of neighboring stops with travel time
        normalize_weight(nbr_weight_df, stop_name, G)
        # performing a roullte-wheel selection (probability-based random selection) using flow weight
        stop_row = nbr_weight_df.sample(n=1, weights='total_flow')
        weight_df.drop(stop_row.index, inplace=True)
        tmp_travel_time = G.edges[stop_name, stop_row['station_name'].values[0]][
            'travel_time']
        longest_time += tmp_travel_time
        stop_row.loc[:, 'travel_time'] = tmp_travel_time  # stop_row is a dataframe
        route.append(stop_row)
        stop_name = stop_row.station_name.values[0]
        out_nbrs = list(G.neighbors(stop_name))

    route_df = pd.concat(route)
    if len(route) > min_stop_num:
        route_df['new_route'] = str(com_id) + '_' + str(route_id)
        route_df['sequence'] = route_df.reset_index(drop=True).index
        return route_df
    else:
        return []


def one_route_set(G, weight_df, route_num, com_id):
    """
    G - truncated graph
    df_terminal
    weight_df
    """
    route_id = 1
    route_set = []
    valid_route = False
    route_set_tmp_df = []
    df_terminal = weight_df.copy(deep=True)
    while len(
            route_set) < route_num:  # repeat until required number of routes in one route have been generated

        weight_df_c = weight_df.copy(deep=True)
        one_route_df = one_route(df_terminal, weight_df_c, G, com_id, route_id)
        if len(one_route_df) > 1:
            route_set.append(one_route_df)
            route_id += 1
    return route_set


def graph_rebuilder(route_ls):
    """
    This func constructs graph based on our generated route set.
    route_ls should ba a list of Dataframes(each is an bus route with stops orderly arranged.)
    """
    G = nx.DiGraph()
    transfer_penalty = 5  # unit in minutes

    for each_graph_df in route_ls:
        row_iterator = each_graph_df.iterrows()
        _, last_row = next(row_iterator)  # first row of the df
        curr_route_id = last_row['new_route']
        if not G.has_node(last_row['station_name']):
            # twin_nodes of original node is a set containing all twin nodes of alternative routes
            G.add_node(last_row['station_name'], long=last_row['long'],
                       lat=last_row['lat'],
                       twin_nodes={last_row['station_name']})
        else:  # if one stop of another route already exist, it's a transfer stop, and split it
            twin_node_id = last_row['station_name'] * 10000 + random.randint(1, 999)

            G.add_node(twin_node_id, long=last_row['long'], lat=last_row['lat'],
                       twin_nodes={twin_node_id})
            # add the twin node of the original node
            G.nodes[last_row['station_name']]['twin_nodes'].add(twin_node_id)
            # add bidirectional edges between a node pair of a transfer stop
            G.add_edge(last_row['station_name'], twin_node_id,
                       travel_time=transfer_penalty,
                       new_route='trans')
            G.add_edge(twin_node_id, last_row['station_name'],
                       travel_time=transfer_penalty,
                       new_route='trans')

            last_row['station_name'] = twin_node_id

        for _, row in row_iterator:  # starting from the 2nd node
            curr_stop_name = row['station_name']
            if not G.has_node(curr_stop_name):
                G.add_node(curr_stop_name, long=row['long'], lat=row['lat'],
                           twin_nodes={curr_stop_name})
            else:
                twin_node_id = curr_stop_name * 10000 + random.randint(1, 999)
                G.add_node(twin_node_id, long=row['long'], lat=row['lat'],
                           twin_nodes={twin_node_id})
                G.nodes[curr_stop_name]['twin_nodes'].add(twin_node_id)

                G.add_edge(curr_stop_name, twin_node_id,
                           travel_time=transfer_penalty,
                           new_route='trans')
                G.add_edge(twin_node_id, curr_stop_name,
                           travel_time=transfer_penalty,
                           new_route='trans')

                curr_stop_name = twin_node_id
                row['station_name'] = twin_node_id

            travel_time = row['travel_time']
            if last_row['station_name'] != curr_stop_name:
                G.add_edge(last_row['station_name'], curr_stop_name,
                           travel_time=travel_time,
                           new_route=curr_route_id)
                G.add_edge(curr_stop_name, last_row['station_name'],
                           travel_time=travel_time,
                           new_route=curr_route_id)

            last_row = row

    return G


def fitness(G, df_od):
    """
    G is the graph rebuilt from the new route set
    df_od is the df of OD flow within the current cluster. O and D are both in the cluster
    route_set is the newly generated route set.
    """
    transfer_penalty = 5  # unit:min
    inaccessibility_penalty = 50 + 24.74  # unit:min
    total_travel_cost = 0
    for each in df_od[['o_station_name', 'd_station_name', 'flow']].values:
        if G.has_node(each[0]) and G.has_node(each[1]):
            if nx.has_path(G, each[0], each[1]):  # if have path
                # if both start & end terminals do not have twin stops
                if (len(G.nodes[each[0]]['twin_nodes'])) == 1 and (
                len(G.nodes[each[1]]['twin_nodes'])) == 1:
                    total_travel_cost += nx.shortest_path_length(G, source=each[0],
                                                                 target=each[1],
                                                                 weight='travel_time') * \
                                         each[2]
                elif (len(G.nodes[each[0]]['twin_nodes']) > 1) and (
                len(G.nodes[each[1]]['twin_nodes'])) == 1:
                    curr_travel_cost = min([
                        nx.shortest_path_length(G, source=each_source, target=each[1],
                                                weight='travel_time') for
                        each_source in G.nodes[each[0]]['twin_nodes']])

                    total_travel_cost += curr_travel_cost * each[2]
                elif (len(G.nodes[each[0]]['twin_nodes']) == 1) and (
                len(G.nodes[each[1]]['twin_nodes'])) > 1:
                    curr_travel_cost = min([
                        nx.shortest_path_length(G, source=each[0], target=each_target,
                                                weight='travel_time') for
                        each_target in G.nodes[each[1]]['twin_nodes']])

                    total_travel_cost += curr_travel_cost * each[2]
                else:
                    curr_travel_cost = min([
                        nx.shortest_path_length(G, source=each_source, target=each_target,
                                                weight='travel_time') for
                        each_source in G.nodes[each[0]]['twin_nodes'] for
                        each_target in G.nodes[each[1]]['twin_nodes']])

                    total_travel_cost += curr_travel_cost * each[2]
            else:
                total_travel_cost += inaccessibility_penalty * each[2]

        else:
            total_travel_cost += inaccessibility_penalty * each[2]
    return total_travel_cost


def multi_optimizer(params):
    start_time_child = time.time()
    com_id = params[0]
    route_num = params[1]
    sg = nx.from_pandas_edgelist(params[2], 'source', 'target', ['travel_time'],
                                 create_using=nx.Graph())
    com_stop_merged_df = params[3]
    com_terminal_weight_df = params[4]
    cost_attr = params[5]
    df_com_od = params[6]

    population_trail = 300
    population = []
    tries = 0

    while tries <= population_trail:

        route_set = one_route_set(sg, com_stop_merged_df, route_num, com_id)

        if route_set is None:
            continue

        population.append(route_set)

        tries += 1

    fitness_selector_df = pd.DataFrame({'fitness': [
        fitness(graph_rebuilder(each_route_set), df_com_od) for each_route_set in
        population]})
    # last_min_fitness = fitness_selector_df.min()[0]

    curr_best_solution = population[fitness_selector_df.idxmin()[0]]

    return pd.concat(curr_best_solution, ignore_index=True)


if __name__ == '__main__':

    params_type = 'GCH'

    with open('./tmp_results/tmp_parameters_mumford3_{}.pkl'.format(params_type), 'rb') as f:
        params = pickle.load(f)

    num_cores = multiprocessing.cpu_count() - 1

    print('start multi-optimization for clusters')
    print('==================================')
    start_optimization = time.time()
    with multiprocessing.Pool(num_cores) as pool:
        opt_route_set_ls = pool.map(multi_optimizer, params, chunksize=1)
        pool.close()
        pool.join()

        print('The multi-optimization for each cluster took %4.3f minutes' % (
                (time.time() - start_optimization) / 60))
        print('==================================')
        opt_route_set_ls = [each_routes_com for each_routes_com in opt_route_set_ls if
                            each_routes_com is not None]

    pd.concat(opt_route_set_ls).to_csv(
        './tmp_results/coms_optimized_routes_mumford3_{}.csv'.format(params_type),
        encoding='utf-8-sig')
