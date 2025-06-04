from ast import literal_eval
import pandas as pd
from ast import literal_eval
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import KMeans
import numpy as np

import json
import urllib
import math

x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 偏心率平方


class Geocoding:
    def __init__(self, api_key):
        self.api_key = api_key

    def geocode(self, address):
        """
        利用高德geocoding服务解析地址获取位置坐标
        :param address:需要解析的地址
        :return:
        """
        geocoding = {'s': 'rsv3',
                     'key': self.api_key,
                     'city': '全国',
                     'address': address}
        geocoding = urllib.urlencode(geocoding)
        ret = urllib.urlopen("%s?%s" % ("http://restapi.amap.com/v3/geocode/geo", geocoding))

        if ret.getcode() == 200:
            res = ret.read()
            json_obj = json.loads(res)
            if json_obj['status'] == '1' and int(json_obj['count']) >= 1:
                geocodes = json_obj['geocodes'][0]
                lng = float(geocodes.get('location').split(',')[0])
                lat = float(geocodes.get('location').split(',')[1])
                return [lng, lat]
            else:
                return None
        else:
            return None


def gcj02_to_bd09(lng, lat):
    """
    火星坐标系(GCJ-02)转百度坐标系(BD-09)
    谷歌、高德——>百度
    :param lng:火星坐标经度
    :param lat:火星坐标纬度
    :return:
    """
    z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * x_pi)
    theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * x_pi)
    bd_lng = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return [bd_lng, bd_lat]


def bd09_to_gcj02(bd_lon, bd_lat):
    """
    百度坐标系(BD-09)转火星坐标系(GCJ-02)
    百度——>谷歌、高德
    :param bd_lat:百度坐标纬度
    :param bd_lon:百度坐标经度
    :return:转换后的坐标列表形式
    """
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
    gg_lng = z * math.cos(theta)
    gg_lat = z * math.sin(theta)
    return [gg_lng, gg_lat]


def wgs84_to_gcj02(lng, lat):
    """
    WGS84转GCJ02(火星坐标系)
    :param lng:WGS84坐标系的经度
    :param lat:WGS84坐标系的纬度
    :return:
    """
    if out_of_china(lng, lat):  # 判断是否在国内
        return [lng, lat]
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [mglng, mglat]


def gcj02_to_wgs84(lng, lat):
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """
    if out_of_china(lng, lat):
        return [lng, lat]
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def bd09_to_wgs84(bd_lon, bd_lat):
    lon, lat = bd09_to_gcj02(bd_lon, bd_lat)
    return gcj02_to_wgs84(lon, lat)


def wgs84_to_bd09(lon, lat):
    lon, lat = wgs84_to_gcj02(lon, lat)
    return gcj02_to_bd09(lon, lat)


def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def out_of_china(lng, lat):
    """
    判断是否在国内，不在国内不做偏移
    :param lng:
    :param lat:
    :return:
    """
    return not (lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55)
  

# 将字符串类型的列表转换为真正的列表对象
def safe_literal_eval(s):
    try:
        return literal_eval(s.replace(';', ','))  # 处理可能存在的分号分隔符
    except:
        return []

df1 = pd.read_csv("/home/users/lichuan/shared/jld2csv/trips_150103.jld2_trips.csv")
df2 = pd.read_csv("/home/users/lichuan/shared/jld2csv/trips_150104.jld2_trips.csv")
df3 = pd.read_csv("/home/users/lichuan/shared/jld2csv/trips_150105.jld2_trips.csv")
df4 = pd.read_csv("/home/users/lichuan/shared/jld2csv/trips_150106.jld2_trips.csv")
df5 = pd.read_csv("/home/users/lichuan/shared/jld2csv/trips_150107.jld2_trips.csv")

df1["lon"] = df1["lon"].apply(safe_literal_eval)
df1["lat"] = df1["lat"].apply(safe_literal_eval)
df2["lon"] = df2["lon"].apply(safe_literal_eval)
df2["lat"] = df2["lat"].apply(safe_literal_eval)
df3["lon"] = df3["lon"].apply(safe_literal_eval)
df3["lat"] = df3["lat"].apply(safe_literal_eval)
df4["lon"] = df4["lon"].apply(safe_literal_eval)
df4["lat"] = df4["lat"].apply(safe_literal_eval)
df5["lon"] = df5["lon"].apply(safe_literal_eval)
df5["lat"] = df5["lat"].apply(safe_literal_eval)

# 处理单行数据的函数
def convert_coords(row):
    converted_lons = []
    converted_lats = []
    for lon_val, lat_val in zip(row['lon'], row['lat']):
        new_coords = wgs84_to_gcj02(lon_val, lat_val)
        converted_lons.append(new_coords[0])
        converted_lats.append(new_coords[1])
    return pd.Series({
        'gcj_lon': converted_lons,
        'gcj_lat': converted_lats
    })
    
# 应用转换到DataFrame
df1[['gcj_lon', 'gcj_lat']] = df1.apply(convert_coords, axis=1)
df2[['gcj_lon', 'gcj_lat']] = df2.apply(convert_coords, axis=1)
df3[['gcj_lon', 'gcj_lat']] = df3.apply(convert_coords, axis=1)
df4[['gcj_lon', 'gcj_lat']] = df4.apply(convert_coords, axis=1)
df5[['gcj_lon', 'gcj_lat']] = df5.apply(convert_coords, axis=1)

df1 = df1.rename(columns={
    'lon': 'wgs_lon',
    'lat': 'wgs_lat'
})

# 重命名转换后的列
df1 = df1.rename(columns={
    'gcj_lon': 'lon',
    'gcj_lat': 'lat'
})

df2 = df2.rename(columns={
    'lon': 'wgs_lon',
    'lat': 'wgs_lat'
})
df2 = df2.rename(columns={
    'gcj_lon': 'lon',
    'gcj_lat': 'lat'
})

df3 = df3.rename(columns={
    'lon': 'wgs_lon',
    'lat': 'wgs_lat'
})

df3 = df3.rename(columns={
    'gcj_lon': 'lon',
    'gcj_lat': 'lat'
})

df4 = df4.rename(columns={
    'lon': 'wgs_lon',
    'lat': 'wgs_lat'
})

df4 = df4.rename(columns={
    'gcj_lon': 'lon',
    'gcj_lat': 'lat'
})

df5 = df5.rename(columns={
    'lon': 'wgs_lon',
    'lat': 'wgs_lat'
})

df5 = df5.rename(columns={
    'gcj_lon': 'lon',
    'gcj_lat': 'lat'
})

# df1.to_csv('gcj_trips_150103.jld2_trips.csv', index=False)
# df2.to_csv('gcj_trips_150104.jld2_trips.csv', index=False)
# df3.to_csv('gcj_trips_150105.jld2_trips.csv', index=False)
# df4.to_csv('gcj_trips_150106.jld2_trips.csv', index=False)
# df5.to_csv('gcj_trips_150107.jld2_trips.csv', index=False)