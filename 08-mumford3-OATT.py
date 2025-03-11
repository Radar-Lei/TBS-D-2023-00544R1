"""
Estimation of the optimal average travel time(OATT) for mumford3 network.
OATT is also the lower bound of the TND problem for mumford3 network.
"""
import networkx as nx
import pandas as pd
import multiprocessing
import time
import numpy as np

def fitness(params):
    """
    G is the graph rebuilt from the new route set
    df_od is the df of OD flow within the current cluster. O and D are both in the cluster
    route_set is the newly generated route set.
    """
    df_od, links_df = params[0], params[1]
    G = nx.from_pandas_edgelist(links_df, source='from', target='to', edge_attr=['travel_time'],create_using=nx.DiGraph())


    total_travel_cost = 0 # unit in minutes
    for each in df_od[['o_station_name', 'd_station_name', 'flow']].values:
        if G.has_node(each[0]) and G.has_node(each[1]):
            if nx.has_path(G, each[0], each[1]):  # if have path
                total_travel_cost += nx.shortest_path_length(G, source=each[0], target=each[1], weight='travel_time') * each[2]
    return total_travel_cost
                
if __name__ == '__main__':
    start = time.time()
    df_links = pd.read_csv('./data/mumford3_links.txt')
    df_nodes = pd.read_csv('./data/mumford3_nodes.txt')
    df_od = pd.read_csv('./data/mumford3_demand.txt')

    df_od = pd.merge(df_od, df_nodes[['id', 'lat', 'lon']], how='inner', left_on='from', right_on='id')
    df_od = pd.merge(df_od, df_nodes[['id', 'lat', 'lon']], how='inner', left_on='to',
                     right_on='id')

    df_od = df_od[['from', 'to', 'demand', 'lat_x', 'lon_x', 'lat_y', 'lon_y']].rename(
        columns={'from': 'o_station_name', 'to': 'd_station_name', 'demand': 'flow', 'lat_x': 'o_lat',
                 'lon_x': 'o_long',
                 'lat_y': 'd_lat', 'lon_y': 'd_long'})

    cores = multiprocessing.cpu_count() - 1
    partitions = cores
    df_od_split = np.array_split(df_od, partitions)
    params = []
    for i in range(partitions):
        params.append([df_od_split[i], df_links])

    with multiprocessing.Pool(cores) as pool:
        # fitness_ls, df_od_ls = zip(*pool.map(fitness, params))
        fitness_ls = pool.map(fitness, params)
        pool.close()
        pool.join()

    print('The evaluation for OATT took %4.3f minutes' % ((time.time() - start) / 60))
    print('The OATT for the optimized_route_set is: {}'.format(sum(fitness_ls) / df_od.flow.sum()))
    