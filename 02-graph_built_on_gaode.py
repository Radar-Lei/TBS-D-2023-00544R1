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
import threading


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


def one_route(G, weight_df, df_terminal, edge_weight, route_num, com_id):
    """
    df_terminal contains the dataframe with in-out-flow weights and coordinates for terminal stops.
    weight_df is the dataframe for all bus stops with weights and coordinates.
    G is graph truncated by the above alog.
    """
    route_set = []
    routes = []  # store
    max_stop_num = 60
    min_stop_num = 10
    max_travel_time = 70  # unit: minute
    min_travel_time = 20
    waiting_time_each_stop = 0.5  # unit: minute, the time a bus stop spend waiting at each bus station

    # need station_name as the index of the Dataframe
    weight_df.set_index('station_name', drop=False, inplace=True)

    if (len(df_terminal) == 0):
        return []

    while (len(routes) <= route_num * 2) and (len(df_terminal) >= 2):
        # start terminal
        o_stop_row = df_terminal.sample(n=1, weights='total_flow').copy()  # stop_row is a dataframe
        o_stop_row.loc[:, 'travel_time'] = 0
        o_stop_name = o_stop_row.station_name.values[0]
        # route.append(o_stop_row)

        df_terminal.drop(o_stop_row.index, inplace=True)

        # end terminal
        d_stop_row = df_terminal.sample(n=1, weights='total_flow').copy()
        d_stop_name = d_stop_row.station_name.values[0]

        # tmp_paths = nx.all_simple_paths(sg, source=o_stop_name, target=d_stop_name, cutoff=max_stop_num)
        # tmp_path = next(tmp_paths)
        # print(path_weight(G, tmp_path, weight=edge_weight))
        if nx.has_path(G, o_stop_name, d_stop_name):
            for tmp_path in k_shortest_paths(G, o_stop_name, d_stop_name, random.randrange(5, 10)):
                if (len(tmp_path) >= min_stop_num) and (
                        min(min_travel_time, max_travel_time) < path_weight(G, tmp_path, weight=edge_weight) + (
                        len(tmp_path) - 2) * waiting_time_each_stop < max(
                    min_travel_time,
                    max_travel_time)) and (len(tmp_path) <= max_stop_num):
                    tmp_path_df = weight_df.loc[tmp_path, :]
                    tmp_path_df['travel_time'] = 0.01
                    row_iter = tmp_path_df.iterrows()
                    _, last_row = next(row_iter)
                    for each_index, each_row in row_iter:
                        tmp_path_df.at[each_index, 'travel_time'] = \
                            G.edges[last_row['station_name'], each_row['station_name']]['travel_time']
                        last_row = each_row

                    routes.append(tmp_path_df)

    if len(routes) >= route_num:
        routes_selector_df = pd.DataFrame({'total_flow': [each_df['total_flow'].sum() for each_df in routes]})
        for each_idx in routes_selector_df.sample(n=route_num, weights='total_flow').index.values:
            route_set.append(routes[each_idx])
        route_id = 1
        for each_df in route_set:
            each_df.reset_index(drop=True, inplace=True)
            each_df['sequence'] = each_df.index.values
            each_df['new_route'] = str(com_id) + '_' + str(route_id)
            route_id += 1

        return route_set

    else:
        route_id = 1
        for each_df in route_set:
            each_df.reset_index(drop=True, inplace=True)
            each_df['sequence'] = each_df.index.values
            each_df['new_route'] = str(com_id) + '_' + str(route_id)
            route_id += 1
        return route_set


def graph_rebuilder(route_ls):
    """
    This func constructs graph based on our generated route set.
    route_ls should ba a list of Dataframes(each is an bus route with stops orderly arranged.)
    """
    G = nx.DiGraph()

    for each_graph_df in route_ls:
        row_iterator = each_graph_df.iterrows()
        _, last_row = next(row_iterator)  # first row of the df
        curr_route_id = last_row['new_route']
        G.add_node(last_row['station_name'], long=last_row['long'], lat=last_row['lat'], new_route={curr_route_id})
        for _, row in row_iterator:
            curr_stop_name = row['station_name']
            if not G.has_node(curr_stop_name):
                G.add_node(curr_stop_name, long=row['long'], lat=row['lat'], new_route={curr_route_id})
            else:
                G.nodes[curr_stop_name]['new_route'].add(curr_route_id)

            travel_time = row['travel_time']
            if last_row['station_name'] != curr_stop_name:
                if not G.has_edge(last_row['station_name'], curr_stop_name):
                    G.add_edge(last_row['station_name'], curr_stop_name, travel_time=travel_time,
                               new_route={curr_route_id})
                else:
                    G.edges[last_row['station_name'], curr_stop_name]['new_route'].add(curr_route_id)
            last_row = row

    return G


def fitness(G, df_od, route_set):
    """
    G is the graph rebuilt from the new route set
    df_od is the df of OD flow within the current cluster. O and D are both in the cluster
    route_set is the newly generated route set.
    """
    transfer_penalty = 20  # unit:min
    inaccessibility_penalty = 50  # unit:min
    total_travel_cost = 0
    for each in df_od[['o_station_name', 'd_station_name', 'flow']].values:
        if G.has_node(each[0]) and G.has_node(each[1]):  # if O and D are in the graph built with new route set
            if nx.has_path(G, each[0], each[1]):  # if have path
                travel_time = []
                o_tmp_ls = [each_route for each_route in route_set if each[0] in each_route.station_name.values]
                for each_o_tmp_df in o_tmp_ls:
                    if each[1] in each_o_tmp_df.station_name.values:
                        o_index = each_o_tmp_df[each_o_tmp_df.station_name == each[0]].sequence.values[0] + 1
                        d_index = each_o_tmp_df[each_o_tmp_df.station_name == each[1]].sequence.values[0] + 1
                        if o_index < d_index:
                            tmp_travel_time = each_o_tmp_df.iloc[o_index:d_index, :].travel_time.sum() * each[2]
                            travel_time.append(tmp_travel_time)

                if len(travel_time) > 0:  # if direct passenger flow
                    total_travel_cost += min(travel_time)
                else:  #
                    total_travel_cost += (nx.shortest_path_length(G, each[0], each[1],
                                                                  weight='travel_time') + transfer_penalty) * each[2]
            else:
                total_travel_cost += inaccessibility_penalty * each[2]
        else:
            total_travel_cost += inaccessibility_penalty * each[2]
    return total_travel_cost


def intra_swap_constraints(route_new_df, route_df, route_id):
    max_stop_num = 60
    min_stop_num = 10
    max_travel_time = 60  # unit: minute
    min_travel_time = 20

    if (len(route_new_df) >= min_stop_num) and (
            min(min_travel_time, max_travel_time) < route_new_df.travel_time.sum() < max(
        min_travel_time,
        max_travel_time)) and (len(route_new_df) <= max_stop_num):
        route_new_df['new_route'] = route_id
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
    np.random.shuffle(parent)  # randomly select two routes (df) from the parent and drop them from it.
    while (not succ_swap_flag) and (len(parent) >= 2):
        routes, parent = parent[:2], parent[2:]
        route_1, route_2 = routes[0], routes[1]
        # routes = random.sample(a, 2)
        # before identifying common stops of these routes, we first need to remove head and tails
        h, t = 1, 1
        stop_array_1 = route_1.station_name.values
        stop_array_2 = route_2.station_name.values
        # remove first and last stops (terminals) as they being crossover site would be invalid.
        stop_array_1 = stop_array_1[h:len(stop_array_1) - t]
        stop_array_2 = stop_array_2[h:len(stop_array_2) - t]
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
            route_id_1 = route_1.iloc[0, :].new_route
            route_id_2 = route_2.iloc[0, :].new_route
            route_1_new = pd.concat([route_1.iloc[:idx_1], route_2.iloc[idx_2:]]).reset_index(drop=True)
            route_2_new = pd.concat([route_2.iloc[:idx_2], route_1.iloc[idx_1:]]).reset_index(drop=True)

            route_1, flag1 = intra_swap_constraints(route_1_new, route_1, route_id_1)
            route_2, flag2 = intra_swap_constraints(route_2_new, route_2, route_id_2)
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

class FitnessThread(threading.Thread):
    def __init__(self, func, args, ):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.result = self.func(self.args[0],self.args[1])

def multi_fitness(df_com_od, population):

    fitness_selector_df = pd.DataFrame({'fitness': [fitness(graph_rebuilder(each_route_set), df_com_od, each_route_set)
                                                    for each_route_set in population]})

    return fitness_selector_df

def crossover(population, df_com_od, route_num, iter, com_id):
    population_size = len(population)
    # threading begin for speed up fitness evaluation
    threads = []
    population_split = np.array_split(population, 20)
    for i in range(len(population_split)):
        t = FitnessThread(multi_fitness, (df_com_od, population_split[i]))
        threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # fitness_selector_df = pd.DataFrame({'fitness': [fitness(graph_rebuilder(each_route_set), df_com_od, each_route_set)
    #                                                 for each_route_set in population]})

    fitness_selector_df = pd.concat([each_thread.result for each_thread in threads])

    ### treading end ##################################

    last_min_fitness = fitness_selector_df.min()[0]

    if math.isnan(last_min_fitness):
        print(population)

    curr_best_solution = population[fitness_selector_df.idxmin()[0]]
    original_cost = last_min_fitness

    print('initial min travel cost of cluster {} is {} minutes with population size: {}'.format(com_id,
                                                                                                int(last_min_fitness),
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
    each_iter_time = time.time()
    # when iteration reach iter or the process converges or the process reach the time limitation (unit: second),
    # we then break the loop and return the best solution for the current cluster.
    while (iteration <= iter) and (convergence < 30) and ((time.time() - each_iter_time) <= 10 * 3600):
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

            # intta-route crossover with multiplied parents (m copies)
            while m > 0:
                child_1 = intra_crossover(parent_1)
                children.append(child_1)
                child_2 = intra_crossover(parent_2)
                children.append(child_2)
                m -= 1

        population.extend(children)

        # threading begin for speed up fitness evaluation
        threads = []
        population_split = np.array_split(children, 20)

        for i in range(len(population_split)):
            t = FitnessThread(multi_fitness, (df_com_od, population_split[i]))
            threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # fitness_selector_children_df = pd.DataFrame(
        #     {'fitness': [fitness(graph_rebuilder(each_route_set), df_com_od, each_route_set)
        #                  for each_route_set in children]})

        fitness_selector_children_df = pd.concat([each_thread.result for each_thread in threads])

        ### treading end ##################################

        fitness_selector_df = pd.concat([fitness_selector_df, fitness_selector_children_df], ignore_index=True)

        curr_min_fitness = fitness_selector_df.min()[0]
        cost_reduction = last_min_fitness - curr_min_fitness

        if cost_reduction > 10:
            convergence = 0
        else:
            convergence += 1

        iteration += 1

        new_population = []  # extend the population of last generation with children

        # preserve the size of population, select "population_size - 1" solutions from the extended_population
        # to make sure the fitness always decrease, we need to make sure the best solution is in the population
        fitness_selector_resample_df = fitness_selector_df.sample(n=population_size - 1, weights=(
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

        if cost_reduction > last_min_fitness * 0.001:  # only print when the deduction is above 2% of the last_min_fitness
            print('current cluster ID: {} with original Min travel cost {} minutes'.format(com_id, int(original_cost)))
            print(
                'current min travel time: {} minutes and last min travel cost: {} minutes'.format(int(curr_min_fitness),
                                                                                                  int(last_min_fitness)))
            print('travel cost deduction: {} minutes '.format(int(cost_reduction)))
            print('The iteration took %4.3f seconds' % (time.time() - each_iter_time))
            print('==================================')

        last_min_fitness = curr_min_fitness

    curr_best_solution_df = pd.concat(curr_best_solution, ignore_index=True)

    curr_best_solution_df['original_cost'] = original_cost
    curr_best_solution_df['optimized'] = last_min_fitness

    return curr_best_solution_df



def multi_optimizer(params):
    com_id = params[0]
    route_num = params[1]
    sg = nx.from_pandas_edgelist(params[2], 'source', 'target', ['travel_time', 'route'], create_using=nx.DiGraph())
    com_stop_merged_df = params[3]
    com_terminal_weight_df = params[4]
    cost_attr = params[5]
    df_com_od = params[6]

    population_trail = 350
    population = []
    tries = 0

    while tries <= population_trail:
        route_set = one_route(sg, com_stop_merged_df.copy(deep=True), com_terminal_weight_df.copy(deep=True),
                              cost_attr, route_num,
                              com_id)

        if len(route_set) > 0:
            population.append(route_set)

        tries += 1

    if len(population) > 2:
        best_solution = crossover(copy.deepcopy(population), df_com_od, route_num, 600, com_id)
        return best_solution

    else:
        return None


if __name__ == '__main__':

    with open('./tmp_results/tmp_parameters.pkl', 'rb') as f:
        params = pickle.load(f)

    print(len(params))

    stop_ls_gd = pd.read_csv('./data/nanjing_station_GD.csv')

    G = init_graph_builder(stop_ls_gd)
    #
    # G_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    # G_df.reset_index(inplace=True)
    # G_df.rename(columns={'index': 'station_name'}, inplace=True)
    #
    # unique_stop_ls_gd = stop_ls_gd.drop_duplicates(subset=['station_name'])
    #
    # stop_weight_df = pd.read_csv('./data/monthly_stop_in_out_weight.csv')
    #
    # stop_weight_dd = dd.from_pandas(stop_weight_df, npartitions=9)
    # stop_weight_dd['station_name'] = stop_weight_dd.stop_name.apply(lambda x: rename_stop(x, unique_stop_ls_gd))
    # stop_weight_df_rename = stop_weight_dd.compute(scheduler='processes')
    #
    # stop_weight_df_rename.loc[:, ['in_flow', 'out_flow', 'total_flow']] = stop_weight_df_rename.loc[:,
    #                                                                       ['in_flow', 'out_flow', 'total_flow']] / 20
    # stop_weight_df_rename = stop_weight_df_rename[
    #     stop_weight_df_rename.groupby('station_name')['total_flow'].transform(max) == stop_weight_df_rename[
    #         'total_flow']]
    #
    # # stop_weight_df_rename.drop(columns = ['stop_name'], axis=1, inplace=True)
    #
    # stop_weight_df_merged = pd.merge(stop_weight_df_rename, G_df, how='right', on=['station_name'])
    # stop_weight_df_merged.drop(columns=['long_x', 'lat_x'], inplace=True)
    # stop_weight_df_merged.rename(columns={'long_y': 'long', 'lat_y': 'lat'}, inplace=True)
    # stop_weight_df_merged.fillna(0, inplace=True)
    #
    # df_od = pd.read_csv('./data/OD_flow_duration.csv')
    # df_od.columns = ['o_stop_name', 'd_stop_name', 'flow', 'o_long', 'o_lat', 'd_long', 'd_lat', 'travel_time']
    #
    # df_od.loc[:, 'flow'] = df_od.loc[:, 'flow'] / 20
    #
    # df_od = pd.merge(df_od, stop_weight_df_rename[['stop_name', 'station_name']].set_index('stop_name'),
    #                  left_on='o_stop_name',
    #                  right_index=True, how='left')
    #
    # df_od.rename(columns={'station_name': 'o_station_name'}, inplace=True)
    #
    # df_od = pd.merge(df_od, stop_weight_df_rename[['stop_name', 'station_name']].set_index('stop_name'),
    #                  left_on='d_stop_name',
    #                  right_index=True, how='left')
    #
    # df_od.rename(columns={'station_name': 'd_station_name'}, inplace=True)
    #
    # stop_weight_df_rename = stop_weight_df_rename.sort_values('total_flow', ascending=False).drop_duplicates(
    #     'station_name')
    # stop_weight_df_rename.reset_index(inplace=True, drop=True)
    #
    # df_od = df_od.sort_values('flow', ascending=False).drop_duplicates(['o_station_name', 'd_station_name'])
    # df_od.dropna(inplace=True)
    # df_od.reset_index(inplace=True, drop=True)
    #
    # terminal_weight_df = stop_weight_df_rename[
    #     stop_weight_df_rename['station_name'].isin(stop_ls_gd.drop_duplicates('line_name').station_name)]
    #
    # del stop_ls_gd, stop_weight_dd, stop_weight_df, unique_stop_ls_gd, G_df
    # gc.collect()

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

        route_num = int(np.round((df_com_od.flow.sum() / total_flow) * 405 * 2))
        if route_num < 1:
            continue

        if len(com_terminal_weight_df) >= 4:
            params.append(
                [com_id, route_num, sg_df, com_stop_merged_df, com_terminal_weight_df, 'travel_time', df_com_od])
        else:
            params.append(
                [com_id, route_num, sg_df, com_stop_merged_df, com_stop_merged_df, 'travel_time', df_com_od])

    print('The parameter preparation took %4.3f minutes' % ((time.time() - start_params) / 60))
    print('==================================')

    print('start multi-optimization for clusters')
    print('==================================')
    start_optimization = time.time()
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(multi_optimizer, params, chunksize=1)
        pool.close()
        pool.join()

        results = [each_routes_com for each_routes_com in results if each_routes_com is not None]

        pd.concat(results).to_csv('./tmp_results/coms_optimized_routes.csv',encoding='utf-8-sig')

        print('The optimization for each cluster took %4.3f minutes' % ((time.time() - start_optimization) / 60))
