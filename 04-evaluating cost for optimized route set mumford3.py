import time
import pandas as pd
import networkx as nx
import numpy as np
import multiprocessing
from networkx.classes.function import path_weight
from itertools import islice
import random



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
            G.add_node(last_row['station_name'], long=last_row['long'], lat=last_row['lat'],
                       twin_nodes={last_row['station_name']})
        else:  # if one stop of another route already exist, it's a transfer stop, and split it
            twin_node_id = last_row['station_name'] * 10000 + random.randint(1, 999)

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
                twin_node_id = curr_stop_name * 10000 + random.randint(1, 999)
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
                G.add_edge(curr_stop_name, last_row['station_name'], travel_time=travel_time,
                           new_route=curr_route_id)

            last_row = row

    return G


def k_shortest_paths(G, source, target, k, weight='travel_time'):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))



def fitness(params):
    """
    G is the graph rebuilt from the new route set
    df_od is the df of OD flow within the current cluster. O and D are both in the cluster
    route_set is the newly generated route set.
    """
    df_od, route_ls = params[0], params[1]
    G = graph_rebuilder(route_ls)

    transfer_penalty = 5  # unit:min
    inaccessibility_penalty = 50 + 24.74  # unit:min
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
                        total_travel_cost += inaccessibility_penalty * each[2]
                        continue

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
                        total_travel_cost += inaccessibility_penalty * each[2]
                        continue

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
                        total_travel_cost += inaccessibility_penalty * each[2]
                        continue

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
                        total_travel_cost += inaccessibility_penalty * each[2]
                        continue

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
    route_set_type = '63.19'
    opt_route_set_df = pd.read_csv('./tmp_results/coms_optimized_routes_mumford3_{}.csv'.format(route_set_type))

    gb = opt_route_set_df.groupby('new_route')
    opt_route_set_ls = [gb.get_group(x) for x in gb.groups]
    print('Number of routes generated is {}'.format(len(opt_route_set_ls)))

    # G = graph_rebuilder(opt_route_set_ls)

    df_nodes = pd.read_csv('./data/mumford3_nodes.txt')
    df_od = pd.read_csv('./data/mumford3_demand.txt')

    df_od = pd.merge(df_od, df_nodes[['id', 'lat', 'lon']], how='inner', left_on='from', right_on='id')
    df_od = pd.merge(df_od, df_nodes[['id', 'lat', 'lon']], how='inner', left_on='to',
                     right_on='id')

    df_od = df_od[['from', 'to', 'demand', 'lat_x', 'lon_x', 'lat_y', 'lon_y']].rename(
        columns={'from': 'o_station_name', 'to': 'd_station_name', 'demand': 'flow', 'lat_x': 'o_lat',
                 'lon_x': 'o_long',
                 'lat_y': 'd_lat', 'lon_y': 'd_long'})

    # df_od['travel_cost'] = np.nan
    total_flow = df_od.flow.sum()

    cores = multiprocessing.cpu_count() - 1
    partitions = cores
    df_od_split = np.array_split(df_od, partitions)
    params = []
    for i in range(partitions):
        params.append([df_od_split[i], opt_route_set_ls])

    with multiprocessing.Pool(cores) as pool:
        # fitness_ls, df_od_ls = zip(*pool.map(fitness, params))
        fitness_ls, num_d0_ls, num_d1_ls, num_d2_ls, num_dun_ls = zip(*pool.map(fitness, params))
        pool.close()
        pool.join()
    ATT = sum(fitness_ls) / df_od.flow.sum()
    print('The evaluation took %4.3f minutes' % ((time.time() - start) / 60))
    print('The ATT for the optimized_route_set is: {}'.format(ATT))
    print('d0: {}'.format(sum(num_d0_ls) / total_flow))
    print('d1: {}'.format(sum(num_d1_ls) / total_flow))
    print('d2: {}'.format(sum(num_d2_ls) / total_flow))
    print('dun: {}'.format(sum(num_dun_ls) / total_flow))
    opt_route_set_df.to_csv('./tmp_results/coms_optimized_routes_mumford3_{}_{:.2f}.csv'.format(route_set_type, ATT) , encoding='utf-8-sig')
