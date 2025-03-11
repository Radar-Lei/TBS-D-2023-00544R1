# -*- coding: utf-8 -*-
"""
Created on Thu May 14 07:45:55 2020
This code can be used to crawl bus routes and station data (both directions) for multiple cities
Required parameters to set:
citys = ['taiyuan','datong']  # List of cities
chinese_city_names = ['太原','大同']  # Corresponding Chinese city names
headers = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36' # Browser user-agent
file_path = 'C://Users//xaoxu//Desktop//bus_data//' # Data storage path
@author: xaoxu
"""

import requests
import pandas as pd
import json
import re
import time
from bs4 import BeautifulSoup
import math


# Get initial letters
def getInitial(cityName, headers):
    url = 'https://{}.8684.cn/list1'.format(cityName)
    headers = {'User-Agent': headers}
    data = requests.get(url, headers=headers)
    soup = BeautifulSoup(data.text, 'lxml')
    initial = soup.find_all('div', {'class': 'tooltip-inner'})[3]
    initial = initial.find_all('a')
    ListInitial = []
    for i in initial:
        ListInitial.append(i.get_text())
    return ListInitial


# Crawl bus routes by initial letter from ListInitial
def getLine(cityName, n, headers, lines):
    url = 'https://{}.8684.cn/list{}'.format(cityName, n)
    headers = {'User-Agent': headers}
    data = requests.get(url, headers=headers)
    soup = BeautifulSoup(data.text, 'lxml')
    busline = soup.find('div', {'class': 'list clearfix'})
    busline = busline.find_all('a')
    for i in busline:
        lines.append(i.get_text())


# Coordinate system conversion
x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # Semi-major axis
ee = 0.00669342162296594323  # Flattening


def gcj02towgs84(lng, lat):
    """
    Convert GCJ02 (Mars coordinate system) to WGS84
    :param lng: Longitude in GCJ02
    :param lat: Latitude in GCJ02
    :return: WGS84 coordinates
    """
    if out_of_china(lng, lat):
        return lng, lat
    dlat = transformlat(lng - 105.0, lat - 35.0)
    dlng = transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))

    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))

    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def out_of_china(lng, lat):
    """
    Check if coordinates are within China (no offset adjustment needed if outside)
    :param lng: Longitude
    :param lat: Latitude
    :return: True if outside China
    """
    if lng < 72.004 or lng > 137.8347:
        return True
    if lat < 0.8293 or lat > 55.8271:
        return True
    return False


def coordinates(c):
    lng, lat = c.split(',')
    lng, lat = float(lng), float(lat)
    wlng, wlat = gcj02towgs84(lng, lat)
    return wlng, wlat


# Crawl bus station information
def get_dt(city, line):
    url = 'https://restapi.amap.com/v3/bus/linename?s=rsv3&extensions=all&key=559bdffe35eec8c8f4dae959451d705c&output=json&city={}&offset=2&keywords={}&platform=JS'.format(
        city, line)
    r = requests.get(url).text
    rt = json.loads(r)
    try:
        if rt['buslines']:
            if len(rt['buslines']) == 0:  # 有名称没数据
                print('no data in list..')
            else:
                du = []
                for cc in range(len(rt['buslines'])):
                    dt = {}
                    dt['line_name'] = rt['buslines'][cc]['name']

                    st_name = []
                    st_coords = []
                    st_sequence = []
                    for st in rt['buslines'][cc]['busstops']:
                        st_name.append(st['name'])
                        st_coords.append(st['location'])
                        st_sequence.append(st['sequence'])

                    dt['station_name'] = st_name
                    dt['station_coords'] = st_coords
                    dt['sequence'] = st_sequence
                    du.append(dt)
                dm = pd.DataFrame(du)
                return dm
        else:
            pass
    except:
        print('error..try it again..')
        time.sleep(2)
        get_dt(city, line)


# Get bus route data
def get_line(city, line):
    url = 'https://restapi.amap.com/v3/bus/linename?s=rsv3&extensions=all&key=559bdffe35eec8c8f4dae959451d705c&output=json&city={}&offset=2&keywords={}&platform=JS'.format(
        city, line)
    r = requests.get(url).text
    rt = json.loads(r)
    try:
        if rt['buslines']:
            if len(rt['buslines']) == 0:  # 有名称没数据
                print('no data in list..')
            else:
                du = []
                for cc in range(len(rt['buslines'])):
                    dt = {}
                    dt['line_name'] = rt['buslines'][cc]['name']
                    dt['polyline'] = rt['buslines'][cc]['polyline']
                    du.append(dt)
                dm = pd.DataFrame(du)
                return dm
        else:
            pass
    except:
        print('error..try it again..')
        time.sleep(2)
        get_dt(city, line)


# Main program for crawling bus data
def get_station_line_inf(citys, chinese_city_names, headers, file_path):
    # Get bus route names for each city in citys list
    for city, city_name in zip(citys, chinese_city_names):
        # Create empty list for bus routes
        lines = []
        # Crawl initial letters list
        ListInitial = getInitial(city, headers)
        # Crawl routes by initial letter and add to lines list
        for n in ListInitial:
            getLine(city, n, headers, lines)
        print('Crawling bus route names for {} city...'.format(city_name))
        # File paths
        station_file = file_path + city + '_station.csv'
        line_file = file_path + city + '_line.csv'

        # Crawl station data
        times = 1
        print('Crawling bus station data for {} city...'.format(city_name))
        for sta in lines:
            if times == 1:
                data = get_dt(city_name, sta)
                times += 1
            else:
                dm = get_dt(city_name, sta)
                data = pd.concat([data, dm], ignore_index=True)

        for i in range(data.shape[0]):
            coord_x = []
            coord_y = []
            for j in data['station_coords'][i]:
                coord_x.append(eval(re.split(',', j)[0]))
                coord_y.append(eval(re.split(',', j)[1]))
            a = [[data['line_name'][i]] * len(data['station_coords'][i]), coord_x, coord_y, data['station_name'][i],
                 data['sequence'][i]]
            df = pd.DataFrame(a).T
            if i == 0:
                df1 = df
            else:
                df1 = pd.concat([df1, df], ignore_index=True)
        df1.columns = ['line_name', 'coord_x', 'coord_y', 'station_name', 'sequence']

        # Convert station coordinates to WGS84
        print('Converting station coordinates to WGS84 for {}...'.format(city_name))
        t_x = []
        t_y = []
        for i in range(len(list(df1['coord_x']))):
            [X, Y] = gcj02towgs84(list(df1['coord_x'])[i], list(df1['coord_y'])[i])
            t_x.append(X)
            t_y.append(Y)
        t_x = pd.DataFrame(t_x)
        t_y = pd.DataFrame(t_y)
        df1['coord_x'] = t_x
        df1['coord_y'] = t_y
        df1.to_csv(station_file, index=None, encoding='utf_8_sig')

        # Crawl route data
        times = 1
        print('Crawling bus route data for {} city...'.format(city_name))
        for sta in lines:
            if times == 1:
                data = get_line(city_name, sta)
                times += 1
            else:
                dm = get_line(city_name, sta)
                data = pd.concat([data, dm], ignore_index=True)

        # Convert route coordinates to WGS84
        print('Converting route coordinates to WGS84 for {}...'.format(city_name))
        name = []
        lons = []
        lats = []
        orders = []
        for uu in range(len(data)):
            linestr = [coordinates(c) for c in data['polyline'][uu].split(';')]
            for m in range(len(linestr)):
                name.append(data['line_name'][uu])
                orders.append(m)
                lons.append(linestr[m][0])
                lats.append(linestr[m][1])
        dre = {'line_name': name, 'lon': lons, 'lat': lats, 'orders': orders}
        data = pd.DataFrame(dre)
        data.to_csv(line_file, index=None, encoding='utf_8_sig')


# Set parameters
citys = ['nanjing']  # List of cities
chinese_city_names = ['南京']
headers = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36'  # Browser user-agent
file_path = 'C://Users//great//Desktop//bus_data//'  # Data storage path
get_station_line_inf(citys, chinese_city_names, headers, file_path)
