"""
The graph representing connectivity topology between bus stops are constructed using data scrapping from
GaoDe API.
"""
import time
import pandas as pd
import difflib
# import dask.dataframe as dd
import networkx as nx
# from math import sin, cos, sqrt, atan2, radians
import math
import gc
# from cdlib import algorithms, viz
import numpy as np
from itertools import combinations
# import leidenalg as la
# import igraph as ig
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
            norm_weight = (df.loc[index, 'total_flow'] + df.loc[
                index, 'com_flow']) / tmp_each_time
            if norm_weight > 1:
                df.at[index, 'total_flow'] = norm_weight
            else:
                df.at[index, 'total_flow'] = 1

        # df.at[index, 'route'] = each_route_ids


def one_route(df_terminal, weight_df, G, com_id, route_id, norm_factor):
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
    df_terminal.at[stop_row.index[0], 'total_flow'] = df_terminal.at[
                                                       stop_row.index[0], 'total_flow'] / norm_factor

    stop_row.loc[:, 'travel_time'] = 0
    route.append(stop_row)

    # drop the terminal from weight_df
    weight_df.drop(stop_row.index[0], inplace=True)

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
        weight_df.drop(stop_row.index[0], inplace=True)
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


def one_route_set(G, weight_df,terminal_df, route_num, com_id,norm_factor):
    """
    G - truncated graph
    df_terminal
    weight_df
    """
    route_id = 1
    route_set = []
    valid_route = False
    route_set_tmp_df = []
    df_terminal = terminal_df.copy(deep=True)
    while len(
            route_set) < route_num:  # repeat until required number of routes in one route have been generated

        weight_df_c = weight_df.copy(deep=True)
        one_route_df = one_route(df_terminal, weight_df_c, G, com_id, route_id, norm_factor)
        if len(one_route_df) > 1:
            route_set.append(one_route_df)
            weight_df.loc[one_route_df.index, 'com_flow'] = weight_df.loc[
                                                                one_route_df.index, 'com_flow'].values / norm_factor
            weight_df.loc[one_route_df.index, 'total_flow'] = weight_df.loc[
                                                                  one_route_df.index, 'total_flow'].values / norm_factor
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

                    path = nx.shortest_path(G, source=each[0], target=each[1],
                                            weight='travel_time')
                    travel_time_ls = [G[u][v]['travel_time'] for (u, v) in
                                      zip(path[0:], path[1:])]
                    num_transfer = travel_time_ls.count(transfer_penalty)
                    if num_transfer > 2:
                        total_travel_cost += inaccessibility_penalty * each[2]
                        continue

                    total_travel_cost += nx.path_weight(G, path, weight='travel_time') * \
                                         each[2]

                elif (len(G.nodes[each[0]]['twin_nodes']) > 1) and (
                        len(G.nodes[each[1]]['twin_nodes'])) == 1:
                    path_ls = [
                        nx.shortest_path(G, source=each_source, target=each[1],
                                         weight='travel_time') for
                        each_source in G.nodes[each[0]]['twin_nodes']]

                    path_weight_ls = [nx.path_weight(G, each_path, weight='travel_time')
                                      for each_path in path_ls]
                    curr_travel_cost = min(path_weight_ls)

                    path = path_ls[path_weight_ls.index(curr_travel_cost)]
                    travel_time_ls = [G[u][v]['travel_time'] for (u, v) in
                                      zip(path[0:], path[1:])]
                    num_transfer = travel_time_ls.count(transfer_penalty)
                    if num_transfer > 2:
                        total_travel_cost += inaccessibility_penalty * each[2]
                        continue

                    total_travel_cost += curr_travel_cost * each[2]

                elif (len(G.nodes[each[0]]['twin_nodes']) == 1) and (
                        len(G.nodes[each[1]]['twin_nodes'])) > 1:
                    path_ls = [
                        nx.shortest_path(G, source=each[0], target=each_target,
                                         weight='travel_time') for
                        each_target in G.nodes[each[1]]['twin_nodes']]

                    path_weight_ls = [nx.path_weight(G, each_path, weight='travel_time')
                                      for each_path in path_ls]
                    curr_travel_cost = min(path_weight_ls)

                    path = path_ls[path_weight_ls.index(curr_travel_cost)]
                    travel_time_ls = [G[u][v]['travel_time'] for (u, v) in
                                      zip(path[0:], path[1:])]
                    num_transfer = travel_time_ls.count(transfer_penalty)
                    if num_transfer > 2:
                        total_travel_cost += inaccessibility_penalty * each[2]
                        continue

                    total_travel_cost += curr_travel_cost * each[2]

                else:
                    path_ls = [
                        nx.shortest_path(G, source=each_source, target=each_target,
                                         weight='travel_time') for
                        each_source in G.nodes[each[0]]['twin_nodes'] for
                        each_target in G.nodes[each[1]]['twin_nodes']]

                    path_weight_ls = [nx.path_weight(G, each_path, weight='travel_time')
                                      for each_path in path_ls]
                    curr_travel_cost = min(path_weight_ls)

                    path = path_ls[path_weight_ls.index(curr_travel_cost)]
                    travel_time_ls = [G[u][v]['travel_time'] for (u, v) in
                                      zip(path[0:], path[1:])]
                    num_transfer = travel_time_ls.count(transfer_penalty)
                    if num_transfer > 2:
                        total_travel_cost += inaccessibility_penalty * each[2]
                        continue

                    total_travel_cost += curr_travel_cost * each[2]
            else:
                total_travel_cost += inaccessibility_penalty * each[2]

        else:
            total_travel_cost += inaccessibility_penalty * each[2]
    return total_travel_cost


def re_fitness(params):
    """
    for single-optimization
    G is the graph rebuilt from the new route set
    df_od is the df of OD flow within the current cluster. O and D are both in the cluster
    route_set is the newly generated route set.
    """
    cost_ls = []
    df_od = params[0][1]
    for i in range(len(params)):
        G = graph_rebuilder(params[i][0])
        transfer_penalty = 5  # unit:min
        inaccessibility_penalty = 50 + 24.74  # unit:min
        total_travel_cost = 0
        for each in df_od[['o_station_name', 'd_station_name', 'flow']].values:
            if G.has_node(each[0]) and G.has_node(each[1]):
                if nx.has_path(G, each[0], each[1]):  # if have path
                    # if both start & end terminals do not have twin stops
                    if (len(G.nodes[each[0]]['twin_nodes'])) == 1 and (
                            len(G.nodes[each[1]]['twin_nodes'])) == 1:

                        path = nx.shortest_path(G, source=each[0], target=each[1],
                                                weight='travel_time')
                        travel_time_ls = [G[u][v]['travel_time'] for (u, v) in
                                          zip(path[0:], path[1:])]
                        num_transfer = travel_time_ls.count(transfer_penalty)
                        if num_transfer > 2:
                            total_travel_cost += inaccessibility_penalty * each[2]
                            continue

                        total_travel_cost += nx.path_weight(G, path,
                                                            weight='travel_time') * each[
                                                 2]

                    elif (len(G.nodes[each[0]]['twin_nodes']) > 1) and (
                            len(G.nodes[each[1]]['twin_nodes'])) == 1:
                        path_ls = [
                            nx.shortest_path(G, source=each_source, target=each[1],
                                             weight='travel_time') for
                            each_source in G.nodes[each[0]]['twin_nodes']]

                        path_weight_ls = [
                            nx.path_weight(G, each_path, weight='travel_time') for
                            each_path in path_ls]
                        curr_travel_cost = min(path_weight_ls)

                        path = path_ls[path_weight_ls.index(curr_travel_cost)]
                        travel_time_ls = [G[u][v]['travel_time'] for (u, v) in
                                          zip(path[0:], path[1:])]
                        num_transfer = travel_time_ls.count(transfer_penalty)
                        if num_transfer > 2:
                            total_travel_cost += inaccessibility_penalty * each[2]
                            continue

                        total_travel_cost += curr_travel_cost * each[2]

                    elif (len(G.nodes[each[0]]['twin_nodes']) == 1) and (
                            len(G.nodes[each[1]]['twin_nodes'])) > 1:
                        path_ls = [
                            nx.shortest_path(G, source=each[0], target=each_target,
                                             weight='travel_time') for
                            each_target in G.nodes[each[1]]['twin_nodes']]

                        path_weight_ls = [
                            nx.path_weight(G, each_path, weight='travel_time') for
                            each_path in path_ls]
                        curr_travel_cost = min(path_weight_ls)

                        path = path_ls[path_weight_ls.index(curr_travel_cost)]
                        travel_time_ls = [G[u][v]['travel_time'] for (u, v) in
                                          zip(path[0:], path[1:])]
                        num_transfer = travel_time_ls.count(transfer_penalty)
                        if num_transfer > 2:
                            total_travel_cost += inaccessibility_penalty * each[2]
                            continue

                        total_travel_cost += curr_travel_cost * each[2]

                    else:
                        path_ls = [
                            nx.shortest_path(G, source=each_source, target=each_target,
                                             weight='travel_time') for
                            each_source in G.nodes[each[0]]['twin_nodes'] for
                            each_target in G.nodes[each[1]]['twin_nodes']]

                        path_weight_ls = [
                            nx.path_weight(G, each_path, weight='travel_time') for
                            each_path in path_ls]
                        curr_travel_cost = min(path_weight_ls)

                        path = path_ls[path_weight_ls.index(curr_travel_cost)]
                        travel_time_ls = [G[u][v]['travel_time'] for (u, v) in
                                          zip(path[0:], path[1:])]
                        num_transfer = travel_time_ls.count(transfer_penalty)
                        if num_transfer > 2:
                            total_travel_cost += inaccessibility_penalty * each[2]
                            continue

                        total_travel_cost += curr_travel_cost * each[2]
                else:
                    total_travel_cost += inaccessibility_penalty * each[2]

            else:
                total_travel_cost += inaccessibility_penalty * each[2]
        cost_ls.append(total_travel_cost)
    return cost_ls


def intra_swap_constraints(route_new_df, route_df):
    max_stop_num = 25
    min_stop_num = 12
    # max_travel_time = 60  # unit: minute
    # min_travel_time = 20
    # the third constraint check is repeat stations exist in a route
    if (len(route_new_df) >= min_stop_num) and (len(route_new_df) <= max_stop_num) and (
            len(set(route_new_df.station_name.values)) == len(route_new_df)):
        # route_new_df['new_route'] = route_id
        route_new_df['sequence'] = route_new_df.index
        # successful swap and return a True flag
        return route_new_df, True
    else:
        # failed intra-swap for this route return a False flag
        return route_df, False


def intra_crossover(parent):
    succ_swap_flag = False
    common_stop_frac = 0.7
    child = []
    np.random.shuffle(
        parent)  # randomly select two routes (df) from the parent and drop them from it.
    while (not succ_swap_flag) and (len(parent) >= 2):
        routes, parent = parent[:2], parent[2:]
        route_1, route_2 = routes[0], routes[1]
        # routes = random.sample(a, 2)
        # before identifying common stops of these routes, we first need to remove head and tails
        h, t = 1, 1
        stop_array_1 = route_1.station_name.values
        stop_array_2 = route_2.station_name.values
        # remove first and last stops (terminals) as they being crossover site would be invalid.
        # do not need this step for Mumford3
        # stop_array_1 = stop_array_1[h:len(stop_array_1) - t]
        # stop_array_2 = stop_array_2[h:len(stop_array_2) - t]
        common_stops = np.intersect1d(stop_array_1, stop_array_2)
        if len(common_stops) == 0:  # if no common stops, continue to the next pair of routes.
            child.append(route_1)
            child.append(route_2)
            continue
        elif len(common_stops) > min(len(stop_array_1),
                                     len(stop_array_2)) * common_stop_frac:  # if two routes are too similar, skip.
            child.append(route_1)
            child.append(route_2)
            continue
        else:
            common_stop_name = np.random.choice(common_stops)
            idx_1 = route_1[route_1.station_name == common_stop_name].index[0]
            idx_2 = route_2[route_2.station_name == common_stop_name].index[0]
            # route_id_1 = route_1.iloc[0, :].new_route
            # route_id_2 = route_2.iloc[0, :].new_route
            route_1_new = pd.concat(
                [route_1.iloc[:idx_1], route_2.iloc[idx_2:]]).reset_index(drop=True)
            route_2_new = pd.concat(
                [route_2.iloc[:idx_2], route_1.iloc[idx_1:]]).reset_index(drop=True)

            route_1, flag1 = intra_swap_constraints(route_1_new, route_1)
            route_2, flag2 = intra_swap_constraints(route_2_new, route_2)
            child.append(route_1)
            child.append(route_2)
            # if any of these two routes successfully swapped, we consider the overall swap a success
            if flag1 or flag2:
                succ_swap_flag = True

    if len(parent) > 0:
        child.extend(parent)
        return child
    else:
        return child


def crossover(population, df_com_od, route_num, com_id, iter, conver, max_cpu_time):
    total_flow = df_com_od.flow.sum()

    population_size = len(population)
    fitness_selector_df = pd.DataFrame(
        {'fitness': [fitness(graph_rebuilder(each_route_set), df_com_od)
                     for each_route_set in population]})

    last_min_fitness = fitness_selector_df.min()[0] / total_flow

    if math.isnan(last_min_fitness):
        print(population)

    curr_best_solution = population[fitness_selector_df.idxmin()[0]]
    original_cost = last_min_fitness

    print(
        'initial ATT of cluster {} with {} routes is {} minutes with population size: {}'.format(
            com_id,
            route_num,
            last_min_fitness,
            population_size))
    print('==================================')

    mating_pool = []
    # solutions with small fitness should have more possibility to be chosen
    # add one to avoid the fitness becoming zeros which would lead to an ValueError if all rows in df have fitness of zeros.
    for each_idx in fitness_selector_df.sample(frac=0.3, weights=(
            fitness_selector_df.fitness.max() - fitness_selector_df.fitness + 1)).index:
        mating_pool.append(population[each_idx])

    iteration = 0
    convergence = 0
    com_conver_fitness = []
    each_crossover_time = time.time()
    # when iteration reach iter or the process converges or the process reach the time limitation (unit: second),
    # we then break the loop and return the best solution for the current cluster.
    while (iteration <= iter) and (convergence < conver) and (
            (time.time() - each_crossover_time) <= max_cpu_time * 3600):
        each_iter_time = time.time()
        children = []
        # non-duplicated selection
        while len(mating_pool) >= 2:

            # the num of copies of the parents in the intra-crossover.
            m = 5

            np.random.shuffle(mating_pool)
            parents, mating_pool = mating_pool[:2], mating_pool[2:]
            parent_1, parent_2 = copy.deepcopy(parents[0]), copy.deepcopy(
                parents[1])  # this deep copy might also be functional
            # parent_1, parent_2 =  pickle.loads(pickle.dumps(parents[0])),  pickle.loads(pickle.dumps(parents[1]))

            # inter-route crossover, toss a coine to create a list of 0,1. if 1, inter-route crossover
            uniform_crossover_mask = np.random.randint(2, size=route_num)
            for i in range(len(uniform_crossover_mask)):
                if uniform_crossover_mask[i] == 1:
                    parent_1[i], parent_2[i] = parent_2[i], parent_1[i]

            # intra-route crossover with multiplied parents (m copies)
            if len(curr_best_solution) == 1:
                parent_1 = parent_1 + parent_2
                parent_2 = parent_1
                while m > 0:
                    child_1 = intra_crossover(parent_1)
                    child_2 = intra_crossover(parent_2)
                    child = child_1 + child_2
                    fitness_selector_child_df = pd.DataFrame({'fitness': [
                        fitness(graph_rebuilder([each_route_set]), df_com_od) for
                        each_route_set in child]})
                    best_child = child[fitness_selector_child_df.idxmin()[0]]
                    children.append([best_child])
                    m -= 1
                continue

            while m > 0:
                child_1 = intra_crossover(parent_1)
                children.append(child_1)
                child_2 = intra_crossover(parent_2)
                children.append(child_2)
                m -= 1

        fitness_selector_children_df = pd.DataFrame(
            {'fitness': [fitness(graph_rebuilder(each_route_set), df_com_od)
                         for each_route_set in children]})

        fitness_selector_resample_children_df = fitness_selector_children_df.sample(n=int(population_size * 0.2),
                                                                  weights=(
                                                                          fitness_selector_children_df.fitness.max() - fitness_selector_children_df.fitness + 1))
        new_children = []
        for each_idx in fitness_selector_resample_children_df.index:
            new_children.append(children[each_idx])


        population.extend(new_children)

        fitness_selector_df = pd.concat(
            [fitness_selector_df, fitness_selector_resample_children_df], ignore_index=True)

        curr_min_fitness = fitness_selector_df.min()[0] / total_flow
        cost_reduction = last_min_fitness - curr_min_fitness

        if (cost_reduction/last_min_fitness) > 0.001:
            convergence = 0
        else:
            convergence += 1

        iteration += 1

        new_population = []  # extend the population of last generation with children

        # preserve the size of population, select "population_size - 1" solutions from the extended_population
        # to make sure the fitness always decrease, we need to make sure the best solution is in the population
        fitness_selector_resample_df = fitness_selector_df.sample(n=population_size - 1,
                                                                  weights=(
                                                                          fitness_selector_df.fitness.max() - fitness_selector_df.fitness + 1))

        for each_idx in fitness_selector_resample_df.index:
            new_population.append(population[each_idx])

        # append the best solution to the population
        curr_best_solution = population[fitness_selector_df.idxmin()[0]]
        new_population.append(curr_best_solution)

        fitness_selector_resample_df = fitness_selector_resample_df.append(
            fitness_selector_df.loc[fitness_selector_df.idxmin(), :], ignore_index=True)

        population = new_population
        fitness_selector_df = fitness_selector_resample_df

        mating_pool = []
        for each_idx in fitness_selector_df.sample(frac=0.3, weights=(
                fitness_selector_df.fitness.max() - fitness_selector_df.fitness + 1)).index:
            mating_pool.append(population[each_idx])

        if cost_reduction > last_min_fitness * 0.01:  # only print when the deduction is above 2% of the last_min_fitness
            print(
                'current cluster ID: {} with {} routes and original ATT {} minutes'.format(
                    com_id, route_num,original_cost))
            print(
                'current ATT: {} minutes and last ATT: {} minutes'.format(
                    curr_min_fitness,
                    last_min_fitness))
            print('travel cost deduction: {:.2f}% '.format(
                (cost_reduction / last_min_fitness) * 100))
            print('The iteration took %4.3f seconds' % (time.time() - each_iter_time))
            print('==================================')

        last_min_fitness = curr_min_fitness

        if iteration <= 300:
            com_ATT = last_min_fitness
            com_conver_fitness.append(com_ATT)

    for route_id in range(len(curr_best_solution)):
        curr_best_solution[route_id]['new_route'] = str(com_id) + '_' + str(route_id)

    com_conver_fitness_df = pd.DataFrame({'fitness': com_conver_fitness})
    com_conver_fitness_df['com_id'] = com_id

    curr_best_solution_df = pd.concat(curr_best_solution, ignore_index=True)

    curr_best_solution_df['original_cost'] = original_cost
    curr_best_solution_df['optimized'] = last_min_fitness

    return curr_best_solution_df, population, com_conver_fitness_df, last_min_fitness


def re_crossover(population, df_com_od, route_num, com_id, iter, conver, max_cpu_time):
    total_flow = df_com_od.flow.sum()
    population_size = len(population)

    cores = multiprocessing.cpu_count() - 3
    partitions = cores

    tmp_params = [[each_route_set, df_com_od] for each_route_set in population]
    tmp_params = np.array_split(tmp_params, partitions)

    with multiprocessing.Pool(cores) as pool:
        # fitness_ls, df_od_ls = zip(*pool.map(fitness, params))
        fitness_ls = pool.map(re_fitness, tmp_params)
        pool.close()
        pool.join()
    fitness_ls = np.concatenate(fitness_ls)
    fitness_selector_df = pd.DataFrame({'fitness': fitness_ls})

    # fitness_selector_df = pd.DataFrame({'fitness': [fitness(graph_rebuilder(each_route_set), df_com_od)
    #                                                 for each_route_set in population]})

    last_min_fitness = fitness_selector_df.min()[0] / total_flow

    if math.isnan(last_min_fitness):
        print(population)

    curr_best_solution = population[fitness_selector_df.idxmin()[0]]
    original_cost = last_min_fitness

    print(
        'initial ATT of cluster {} is {} minutes with population size: {}'.format(
            com_id,
            last_min_fitness,
            population_size))
    print('==================================')

    mating_pool = []
    # solutions with small fitness should have more possibility to be chosen
    # add one to avoid the fitness becoming zeros which would lead to an ValueError if all rows in df have fitness of zeros.
    for each_idx in fitness_selector_df.sample(frac=0.3, weights=(
            fitness_selector_df.fitness.max() - fitness_selector_df.fitness + 1)).index:
        mating_pool.append(population[each_idx])

    iteration = 0
    convergence = 0
    each_crossover_time = time.time()
    # when iteration reach iter or the process converges or the process reach the time limitation (unit: second),
    # we then break the loop and return the best solution for the current cluster.
    while (iteration <= iter) and (convergence < conver) and (
            (time.time() - each_crossover_time) <= max_cpu_time * 3600):
        each_iter_time = time.time()
        children = []
        # non-duplicated selection
        while len(mating_pool) >= 2:

            # the num of copies of the parents in the intra-crossover.
            m = 5

            np.random.shuffle(mating_pool)
            parents, mating_pool = mating_pool[:2], mating_pool[2:]
            parent_1, parent_2 = copy.deepcopy(parents[0]), copy.deepcopy(
                parents[1])  # this deep copy might also be functional
            # parent_1, parent_2 =  pickle.loads(pickle.dumps(parents[0])),  pickle.loads(pickle.dumps(parents[1]))

            # inter-route crossover
            uniform_crossover_mask = np.random.randint(2, size=route_num)
            for i in range(len(uniform_crossover_mask)):
                if uniform_crossover_mask[i] == 1:
                    parent_1[i], parent_2[i] = parent_2[i], parent_1[i]

            # intra-route crossover with multiplied parents (m copies)
            if len(curr_best_solution) == 1:
                parent_1 = parent_1 + parent_2
                parent_2 = parent_1
                while m > 0:
                    child_1 = intra_crossover(parent_1)
                    child_2 = intra_crossover(parent_2)
                    child = child_1 + child_2
                    fitness_selector_child_df = pd.DataFrame({'fitness': [
                        fitness(graph_rebuilder([each_route_set]), df_com_od) for
                        each_route_set in child]})
                    best_child = child[fitness_selector_child_df.idxmin()[0]]
                    children.append([best_child])
                    m -= 1
                continue

            while m > 0:
                child_1 = intra_crossover(parent_1)
                children.append(child_1)
                child_2 = intra_crossover(parent_2)
                children.append(child_2)
                m -= 1

        tmp_params = [[each_route_set, df_com_od] for each_route_set in children]
        tmp_params = np.array_split(tmp_params, partitions)

        with multiprocessing.Pool(cores) as pool:
            # fitness_ls, df_od_ls = zip(*pool.map(fitness, params))
            fitness_ls = pool.map(re_fitness, tmp_params)
            pool.close()
            pool.join()
        fitness_ls = np.concatenate(fitness_ls)
        fitness_selector_children_df = pd.DataFrame({'fitness': fitness_ls})

        fitness_selector_resample_children_df = fitness_selector_children_df.sample(n=int(population_size * 0.2),
                                                                  weights=(
                                                                          fitness_selector_children_df.fitness.max() - fitness_selector_children_df.fitness + 1))
        new_children = []
        for each_idx in fitness_selector_resample_children_df.index:
            new_children.append(children[each_idx])


        population.extend(new_children)

        fitness_selector_df = pd.concat(
            [fitness_selector_df, fitness_selector_resample_children_df], ignore_index=True)

        # fitness_selector_children_df = pd.DataFrame(
        #     {'fitness': [fitness(graph_rebuilder(each_route_set), df_com_od)
        #                  for each_route_set in children]})

        curr_min_fitness = fitness_selector_df.min()[0] / total_flow
        cost_reduction = last_min_fitness - curr_min_fitness

        if (cost_reduction/last_min_fitness) > 0.001:
            convergence = 0
        else:
            convergence += 1

        iteration += 1

        new_population = []  # extend the population of last generation with children

        # preserve the size of population, select "population_size - 1" solutions from the extended_population
        # to make sure the fitness always decrease, we need to make sure the best solution is in the population
        fitness_selector_resample_df = fitness_selector_df.sample(n=population_size - 1,
                                                                  weights=(
                                                                          fitness_selector_df.fitness.max() - fitness_selector_df.fitness + 1))

        for each_idx in fitness_selector_resample_df.index:
            new_population.append(population[each_idx])

        # append the best solution to the population
        curr_best_solution = population[fitness_selector_df.idxmin()[0]]
        new_population.append(curr_best_solution)

        fitness_selector_resample_df = fitness_selector_resample_df.append(
            fitness_selector_df.loc[fitness_selector_df.idxmin(), :], ignore_index=True)

        population = new_population
        fitness_selector_df = fitness_selector_resample_df

        mating_pool = []
        for each_idx in fitness_selector_df.sample(frac=0.3, weights=(
                fitness_selector_df.fitness.max() - fitness_selector_df.fitness + 1)).index:
            mating_pool.append(population[each_idx])

        if cost_reduction > last_min_fitness * 0.01:  # only print when the deduction is above 2% of the last_min_fitnes
            print(
                'current cluster ID: {} with original ATT {} minutes'.format(
                    com_id, original_cost))
            print(
                'current ATT: {} minutes and last ATT: {} minutes'.format(
                    curr_min_fitness,
                    last_min_fitness))
            print('travel cost deduction: {:.2f}% '.format(
                (cost_reduction / last_min_fitness) * 100))
            print('The iteration took %4.3f seconds' % (time.time() - each_iter_time))
            print('==================================')

        last_min_fitness = curr_min_fitness

    for route_id in range(len(curr_best_solution)):
        curr_best_solution[route_id]['new_route'] = str(com_id) + '_' + str(route_id)

    curr_best_solution_df = pd.concat(curr_best_solution, ignore_index=True)

    curr_best_solution_df['original_cost'] = original_cost
    curr_best_solution_df['optimized'] = last_min_fitness

    return curr_best_solution_df


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

    population_trail = 500
    iter, conver, max_cpu_time = 720, 200, 2  # max_cpu_time in hours
    norm_factor = 1.2
    population = []
    tries = 0

    while tries <= population_trail:

        route_set = one_route_set(sg, com_stop_merged_df,com_terminal_weight_df, route_num, com_id, norm_factor)

        if route_set is None:
            continue

        population.append(route_set)

        tries += 1

    if len(population) > 2:
        best_solution, current_population, com_conver_fitness_df, curr_min_ATT = crossover(
            copy.deepcopy(population), df_com_od, route_num, com_id, iter, conver,
            max_cpu_time)
        cluster_time = (time.time() - start_time_child) / 60
        print('Computational cost for cluster {} is: {:.2f} minutes'.format(com_id,
                                                                            cluster_time))
        print('==================================')
        return best_solution, current_population, cluster_time, df_com_od, route_num, com_id, com_conver_fitness_df, curr_min_ATT
    else:
        return None


if __name__ == '__main__':
    with open('./tmp_results/tmp_parameters_mumford3_GCH.pkl', 'rb') as f:
        params = pickle.load(f)

    num_cores = multiprocessing.cpu_count() - 3

    num_trail = 1
    for i in range(num_trail):

        print('start multi-optimization for clusters')
        print('==================================')
        start_optimization = time.time()
        with multiprocessing.Pool(num_cores) as pool:
            opt_route_set_ls, population_ls, comp_cost_ls, df_com_od_ls, route_num_ls, com_id_ls, com_conver_fitness_df_ls, min_ATT_ls = zip(
                *pool.map(multi_optimizer, params, chunksize=1))
            pool.close()
            pool.join()

            print('The multi-optimization for each cluster took %4.3f minutes' % (
                    (time.time() - start_optimization) / 60))
            print('==================================')
            opt_route_set_ls = [each_routes_com for each_routes_com in opt_route_set_ls if
                                each_routes_com is not None]
            comp_cost_ls = np.array(
                comp_cost_ls)  # convert the list of computational cost into array

        flow_ls = np.array([each_com_od.flow.sum() for each_com_od in df_com_od_ls])
        total_flow = sum(flow_ls)
        global_ATT = sum((flow_ls / total_flow) * min_ATT_ls)
        print('Curr_global_ATT = {} minutes'.format(global_ATT))
        print('====================================')

        print('start single-optimization for clusters')
        print('====================================')
        start = time.time()
        # keep opt routes of clusters that already converged
        opt_route_df_1st = [opt_route_set_ls[each] for each in
                            np.where(comp_cost_ls < 120)[0]]

        # clusters needed to be re-optimized
        re_id_ls = np.where(comp_cost_ls >= 120)[0]
        re_opt_route_set_ls = []
        for each_id in re_id_ls:
            iter = 200
            conver = 100
            max_cpu_time = 0.5
            start_optimization = time.time()
            re_opt_route_set_ls.append(
                re_crossover(population_ls[each_id], df_com_od_ls[each_id],
                             route_num_ls[each_id], com_id_ls[each_id],
                             iter, conver,
                             max_cpu_time))

        print('The single-optimization for all clusters took %4.3f minutes' % (
                (time.time() - start) / 60))
        print('====================================')
        # pd.concat(results).to_csv('./tmp_results/coms_optimized_routes.csv', encoding='utf-8-sig')
        print('start global fitness evaluation')
        start = time.time()
        opt_route_set_ls = opt_route_df_1st + re_opt_route_set_ls
        # G = graph_rebuilder(opt_route_set_ls)
        pd.concat(opt_route_set_ls).to_csv(
            './tmp_results/coms_optimized_routes_mumford3_{}.csv'.format(round(global_ATT,2)),
            encoding='utf-8-sig')
        pd.concat(com_conver_fitness_df_ls).to_csv(
            './tmp_results/com_conver_fitness_mumford3_{}.csv'.format(round(global_ATT,2)),
            encoding='utf-8-sig')

        # script_descriptor = open("04-evaluating cost for optimized route set mumford3.py")
        # a_script = script_descriptor.read()
        # exec(a_script)
