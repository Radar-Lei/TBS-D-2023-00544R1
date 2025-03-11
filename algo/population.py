from networkx.classes.function import path_weight
from itertools import islice
import networkx as nx
import random
import numpy as np
import pandas as pd

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
    max_travel_time = 60  # unit: minute
    min_travel_time = 20
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
            for tmp_path in k_shortest_paths(G, o_stop_name, d_stop_name, random.randrange(5,10)):
                if (len(tmp_path) >= min_stop_num) and (
                        min(min_travel_time, max_travel_time) < path_weight(G, tmp_path, weight=edge_weight) < max(
                    min_travel_time,
                    max_travel_time)) and (len(tmp_path) <= max_stop_num):
                    tmp_path_df = weight_df.loc[tmp_path, :]
                    tmp_path_df['travel_time'] = np.nan
                    row_iter = tmp_path_df.iterrows()
                    _, last_row = next(row_iter)
                    for each_index, each_row in row_iter:
                        tmp_path_df.at[each_index, 'travel_time'] = G.edges[last_row['station_name'], each_row['station_name']]['travel_time']
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

def population_generator(_):
    com_id = 0
    route_set = one_route(sg, com_stop_merged_df.copy(deep=True), com_terminal_weight_df.copy(deep=True), 'travel_time',
                          route_num,
                          com_id)
    if len(route_set) > 0:
        return route_set
    else:
        return np.nan