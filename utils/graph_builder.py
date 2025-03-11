import pandas as pd
import networkx as nx

def single_day_graph(day):
    if day < 10:
        day = '0' + str(day)
    df_avl = pd.read_csv('./data/avl/nanjing_od_estimation_dbo_avl07{}.csv'.format(day))
    G = nx.DiGraph()

    df_avl.dropna(subset=['long','lat'],inplace=True)
    df_avl['进出站时间'] = pd.to_datetime(df_avl['进出站时间'])
    # group by route id and vehicle id
    gb = df_avl.groupby(['线路名称','车辆编号'])

    for each_group in gb.groups:
        each_group = gb.get_group(each_group)
        each_group = each_group.sort_values(by='进出站时间', ascending=True).head(120)  # sort each group(dataframe) by time and fetching the first 120 records
        row_iterator = each_group.iterrows()
        _, last_row = next(row_iterator)  # take the first row, and the pointer of row_iterator would be at the second row.
        curr_route_id = last_row['线路名称'] # stop name of the first row
        G.add_node(last_row['站点名称'], long=last_row['long'], lat=last_row['lat'], route={curr_route_id})  # add the first stop to the graph
        for _, row in row_iterator:
            curr_stop_name = row['站点名称']
            if not G.has_node(curr_stop_name):
                # add the node into the graph if the node is not in the graph, and add three node attributes
                G.add_node(curr_stop_name, long=row['long'], lat=row['lat'], route={curr_route_id})
            else:
                # if the node is already included in the graph, only append its route id to the existed node in the current graph.
                G.nodes[curr_stop_name]['route'].add(curr_route_id)

            # travel time between two consecutive rows in avl data
            travel_time = (row['进出站时间'] - last_row['进出站时间']).total_seconds() / 60
            if last_row['站点名称'] != curr_stop_name: # if these two consecutive rows are different bus stops
                if not G.has_edge(last_row['站点名称'], curr_stop_name): # if the edge from last row to current row not in the current graph, add it.
                    G.add_edge(last_row['站点名称'], curr_stop_name, travel_time=travel_time,route={curr_route_id})
                else:
                    # if this edge already exists
                    curr_travel_time = G.edges[last_row['站点名称'], curr_stop_name]['travel_time'] # get the travel_time of the edge in the graph
                    G.edges[last_row['站点名称'], curr_stop_name]['travel_time'] = (curr_travel_time + travel_time) / 2 # update the travel_time of the edge by averaging
                    G.edges[last_row['站点名称'], curr_stop_name]['route'].add(curr_route_id) # append route_id to the route_id set of the existing edge
            last_row = row # assign current row to last row and continue the iteration
    return G