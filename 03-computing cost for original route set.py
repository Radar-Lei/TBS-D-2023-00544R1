import time
import pandas as pd
import difflib
import networkx as nx
import math
import numpy as np
import multiprocessing
import ast
import re
import random


def SetConv(txt):
    # convert array of strings into sets for the route ids' stored in the .csv
    return set() if txt == 'set()' else ast.literal_eval(txt)


def string_similar(s1, s2):
    return difflib.SequenceMatcher(lambda x: x == " ", s1, s2).quick_ratio()


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
    route_set = []
    for each_group in gb.groups:
        each_group = gb.get_group(each_group)
        each_group = each_group.sort_values(by='sequence', ascending=True)
        each_group['travel_time'] = np.nan
        each_group = each_group.reset_index(drop=True)
        row_iterator = each_group.iterrows()
        _, last_row = next(
            row_iterator)  # take the first row, and the pointer of row_iterator would be at the second row.
        curr_route_id = last_row['line_name']  # stop name of the first row
        G.add_node(last_row['station_name'], long=last_row['coord_x'], lat=last_row['coord_y'], route={curr_route_id})
        for index, row in row_iterator:
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
            each_group.at[index, 'travel_time'] = travel_time
            if last_row['station_name'] != curr_stop_name:  # if these two consecutive rows are different bus stops
                if not G.has_edge(last_row['station_name'],
                                  curr_stop_name):  # if the edge from last row to current row not in the current graph, add it.
                    G.add_edge(last_row['station_name'], curr_stop_name, travel_time=travel_time, route={curr_route_id})
                else:
                    G.edges[last_row['station_name'], curr_stop_name]['route'].add(
                        curr_route_id)  # append route_id to the route_id set of the existing edge
            last_row = row  # assign current row to last row and continue the iteration

        route_set.append(each_group.drop_duplicates(subset=['sequence'], ignore_index=True))
    return G, route_set


def graph_rebuilder(route_ls):
    """
    This func constructs graph based on our generated route set.
    route_ls should ba a list of Dataframes(each is an bus route with stops orderly arranged.)
    """
    G = nx.DiGraph()
    transfer_penalty = 20  # unit in minutes

    for each_graph_df in route_ls:
        row_iterator = each_graph_df.iterrows()
        _, last_row = next(row_iterator)  # first row of the df
        curr_route_id = last_row['new_route']
        if not G.has_node(last_row['station_name']):
            # twin_nodes of original node is a set containing all twin nodes of alternative routes
            G.add_node(last_row['station_name'], long=last_row['long'], lat=last_row['lat'],
                       twin_nodes={last_row['station_name']})
        else:  # if one stop of another route already exist, it's a transfer stop, and split it
            twin_node_id = last_row['station_name'] + str(random.randint(1, 999))

            G.add_node(twin_node_id, long=last_row['long'], lat=last_row['lat'],
                       twin_nodes={twin_node_id})
            # add the twin node of the original node
            G.nodes[last_row['station_name']]['twin_nodes'].add(twin_node_id)
            # add bidirectional edges between a node pair of a transfer stop
            G.add_edge(last_row['station_name'], twin_node_id, travel_time=transfer_penalty,
                       new_route='trans')
            G.add_edge(twin_node_id, last_row['station_name'], travel_time=transfer_penalty,
                       new_route='trans')

            last_row['station_name'] = twin_node_id

        for _, row in row_iterator:  # starting from the 2nd node
            curr_stop_name = row['station_name']
            if not G.has_node(curr_stop_name):
                G.add_node(curr_stop_name, long=row['long'], lat=row['lat'], twin_nodes={curr_stop_name})
            else:
                twin_node_id = curr_stop_name + str(random.randint(1, 999))
                G.add_node(twin_node_id, long=row['long'], lat=row['lat'], twin_nodes={twin_node_id})
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
                G.add_edge(last_row['station_name'], curr_stop_name, travel_time=travel_time,
                           new_route=curr_route_id)

            last_row = row

    return G


def fitness(params):
    """
    G is the graph rebuilt from the new route set
    df_od is the df of OD flow within the current cluster. O and D are both in the cluster
    route_set is the newly generated route set.
    """
    df_od, route_ls = params[0], params[1]
    G = graph_rebuilder(route_ls)

    transfer_penalty = 20  # unit in minutes
    inaccessibility_penalty = 50 + 25.23  # unit:min
    total_travel_cost = 0
    num_d0 = 0
    num_d1 = 0
    num_d2 = 0
    num_dun = 0

    for each in df_od[['o_station_name', 'd_station_name', 'flow']].values:
        if G.has_node(each[0]) and G.has_node(each[1]):
            if nx.has_path(G, each[0], each[1]):  # if have path
                # if both start & end terminals do not have twin stops
                if (len(G.nodes[each[0]]['twin_nodes'])) == 1 and (len(G.nodes[each[1]]['twin_nodes'])) == 1:

                    path = nx.shortest_path(G, source=each[0], target=each[1], weight='travel_time')
                    travel_time_ls = [G[u][v]['travel_time'] for (u, v) in zip(path[0:], path[1:])]
                    num_transfer = travel_time_ls.count(transfer_penalty)
                    if num_transfer == 0:
                        num_d0 += each[2]
                    elif num_transfer == 1:
                        num_d1 += each[2]
                    elif num_transfer == 2:
                        num_d2 += each[2]
                    else:
                        num_dun += each[2]

                    total_travel_cost += nx.path_weight(G, path, weight='travel_time') * each[2]

                elif (len(G.nodes[each[0]]['twin_nodes']) > 1) and (len(G.nodes[each[1]]['twin_nodes'])) == 1:
                    path_ls = [
                        nx.shortest_path(G, source=each_source, target=each[1], weight='travel_time') for
                        each_source in G.nodes[each[0]]['twin_nodes']]

                    path_weight_ls = [nx.path_weight(G, each_path, weight='travel_time') for each_path in path_ls]
                    curr_travel_cost = min(path_weight_ls)

                    path = path_ls[path_weight_ls.index(curr_travel_cost)]
                    travel_time_ls = [G[u][v]['travel_time'] for (u, v) in zip(path[0:], path[1:])]
                    num_transfer = travel_time_ls.count(transfer_penalty)
                    if num_transfer == 0:
                        num_d0 += each[2]
                    elif num_transfer == 1:
                        num_d1 += each[2]
                    elif num_transfer == 2:
                        num_d2 += each[2]
                    else:
                        num_dun += each[2]

                    total_travel_cost += curr_travel_cost * each[2]

                elif (len(G.nodes[each[0]]['twin_nodes']) == 1) and (len(G.nodes[each[1]]['twin_nodes'])) > 1:
                    path_ls = [
                        nx.shortest_path(G, source=each[0], target=each_target, weight='travel_time') for
                        each_target in G.nodes[each[1]]['twin_nodes']]

                    path_weight_ls = [nx.path_weight(G, each_path, weight='travel_time') for each_path in path_ls]
                    curr_travel_cost = min(path_weight_ls)

                    path = path_ls[path_weight_ls.index(curr_travel_cost)]
                    travel_time_ls = [G[u][v]['travel_time'] for (u, v) in zip(path[0:], path[1:])]
                    num_transfer = travel_time_ls.count(transfer_penalty)
                    if num_transfer == 0:
                        num_d0 += each[2]
                    elif num_transfer == 1:
                        num_d1 += each[2]
                    elif num_transfer == 2:
                        num_d2 += each[2]
                    else:
                        num_dun += each[2]

                    total_travel_cost += curr_travel_cost * each[2]

                else:
                    path_ls = [
                        nx.shortest_path(G, source=each_source, target=each_target, weight='travel_time') for
                        each_source in G.nodes[each[0]]['twin_nodes'] for
                        each_target in G.nodes[each[1]]['twin_nodes']]

                    path_weight_ls = [nx.path_weight(G, each_path, weight='travel_time') for each_path in path_ls]
                    curr_travel_cost = min(path_weight_ls)

                    path = path_ls[path_weight_ls.index(curr_travel_cost)]
                    travel_time_ls = [G[u][v]['travel_time'] for (u, v) in zip(path[0:], path[1:])]
                    num_transfer = travel_time_ls.count(transfer_penalty)
                    if num_transfer == 0:
                        num_d0 += each[2]
                    elif num_transfer == 1:
                        num_d1 += each[2]
                    elif num_transfer == 2:
                        num_d2 += each[2]
                    else:
                        num_dun += each[2]

                    total_travel_cost += curr_travel_cost * each[2]
            else:
                total_travel_cost += inaccessibility_penalty * each[2]
                num_dun += each[2]

        else:
            total_travel_cost += inaccessibility_penalty * each[2]
            num_dun += each[2]
    return total_travel_cost, num_d0, num_d1, num_d2, num_dun


if __name__ == '__main__':
    start = time.time()

    stop_ls_from_data = pd.read_csv('./data/unique_route_stop_from_data.csv')
    stop_ls_gd = pd.read_csv('./data/nanjing_station_GD.csv')
    stop_ls_from_data.drop_duplicates(subset=['route_name'], inplace=True)
    stop_ls_gd.drop_duplicates(subset=['line_name'], inplace=True)
    stop_ls_gd['route_name'] = stop_ls_gd.line_name.map(lambda x: re.sub(r'\(.*?\)', '', x))

    tmp_ls = []
    for _, each_row in stop_ls_from_data.iterrows():
        ind = stop_ls_gd.route_name.apply(lambda x: string_similar(x, each_row)).idxmax()
        route_name = stop_ls_gd.loc[ind, 'route_name']
        index = stop_ls_gd[stop_ls_gd.route_name == route_name].index
        tmp_ls.append(stop_ls_gd.loc[index, 'line_name'])
        stop_ls_gd = stop_ls_gd.drop(index)
    tmp_ls = np.concatenate(tmp_ls)

    stop_ls_gd = pd.read_csv('./data/nanjing_station_GD.csv')

    G, original_route_set = init_graph_builder(stop_ls_gd[stop_ls_gd.line_name.isin(tmp_ls)])

    original_route_set = [each_route_df.rename(columns={'line_name': 'new_route', 'coord_x': 'long', 'coord_y': 'lat'})
                          for each_route_df in original_route_set]

    df_od = pd.read_csv('./data/OD_flow_duration_f.csv')
    cores = multiprocessing.cpu_count() - 1
    partitions = cores
    df_od_split = np.array_split(df_od, partitions)
    params = []
    for i in range(partitions):
        params.append([df_od_split[i], original_route_set])

    with multiprocessing.Pool(cores) as pool:
        # fitness_ls, df_od_ls = zip(*pool.map(fitness, params))
        fitness_ls, num_d0_ls, num_d1_ls, num_d2_ls, num_dun_ls = zip(*pool.map(fitness, params))
        pool.close()
        pool.join()

    ATT = sum(fitness_ls) / df_od.flow.sum()
    print('The evaluation took %4.3f minutes' % ((time.time() - start) / 60))
    print('The ATT for the optimized_route_set is: {}'.format(ATT))
    print('d0: {}'.format(sum(num_d0_ls) / df_od.flow.sum()))
    print('d1: {}'.format(sum(num_d1_ls) / df_od.flow.sum()))
    print('d2: {}'.format(sum(num_d2_ls) / df_od.flow.sum()))
    print('dun: {}'.format(sum(num_dun_ls) / df_od.flow.sum()))
    # opt_route_set_df.to_csv('./tmp_results/coms_optimized_routes_mumford3_%4.3f.csv' % ATT, encoding='utf-8-sig')
