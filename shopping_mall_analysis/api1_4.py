import pandas as pd
from ast import literal_eval
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import KMeans
import numpy as np


# 将字符串类型的列表转换为真正的列表对象
def safe_literal_eval(s):
    try:
        return literal_eval(s.replace(';', ','))  # 处理可能存在的分号分隔符
    except:
        return []


def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两点直线距离

    Args:
        lat1 (float): 纬度1
        lon1 (float): 经度1
        lat2 (float): 纬度2
        lon2 (float): 经度2

    Returns:
        float: 直线距离
    """
    # 将角度转换为弧度
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # 差值
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine 公式
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    # 地球半径（单位：千米）
    R = 6378.137
    distance = R * c
    
    return distance  # 返回两点间的距离（单位：千米）
  
  
def calculate_driving_distance(lon_list, lat_list):
    if len(lon_list) < 2 or len(lat_list) < 2:
        return 0.0
    
    total_distance = 0.0
    for i in range(len(lon_list)-1):
        lat1, lon1 = float(lat_list[i]), float(lon_list[i])
        lat2, lon2 = float(lat_list[i+1]), float(lon_list[i+1])
        total_distance += haversine_distance(lat1, lon1, lat2, lon2)
    
    return round(total_distance, 4)  # 保留4位小数
    
    
def api14_average_driving_distance_and_time():
    # 读取 CSV 文件
    df = pd.read_csv("./example.csv")

    # 转换 tms 列
    df["tms"] = df["tms"].apply(safe_literal_eval)
    df["lon"] = df["lon"].apply(safe_literal_eval)
    df["lat"] = df["lat"].apply(safe_literal_eval)

    # 新增 begin_time 列（取 tms 列表的第一个元素）
    df["begin_time"] = df["tms"].apply(lambda x: x[0] if len(x) > 0 else None)
    df["end_time"] = df["tms"].apply(lambda x: x[len(x)-1] if len(x) > 0 else None)
    df["during_time"] = (df["end_time"] - df["begin_time"])/60
    df["driving_distance_km"] = df.apply(
        lambda row: calculate_driving_distance(row["lon"], row["lat"]), 
        axis=1
    )
    df["average_speed_km_per_hour"] = df["driving_distance_km"] / (df["during_time"]/60)
    
    return (df["driving_distance_km"].sum() / len(df["driving_distance_km"]), df["during_time"].sum()/len(df["during_time"]))


def add_cluster_id(df, position_col='begin_position', k=5):
    """
    为 DataFrame 新增聚类 ID 列
    :param df: 输入 DataFrame, 需包含 position_col 列
    :param position_col: 存储经纬度位置的列名，格式为 [lon, lat]
    :param k: 聚类簇数
    :return: 新增了聚类 ID 列的 DataFrame
    """
    # 提取有效坐标（过滤空值）
    valid_positions = df[position_col].dropna()
    
    # 检查是否有有效数据
    if len(valid_positions) == 0:
        df[f"{position_col.split('_')[0]}_area_id"] = np.nan
        return df
    
    # 转换为二维数组 [[lon1, lat1], [lon2, lat2], ...]
    coordinates = np.array(valid_positions.tolist())
    
    # K-means 聚类（使用经纬度作为二维平面坐标）
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(coordinates)
    
    # 生成簇标签（从1开始，0保留给无效数据）
    labels = kmeans.labels_ + 1
    
    # 创建新列并填充簇ID
    new_col = f"{position_col.split('_')[0]}_area_id"
    df[new_col] = 0  # 默认0表示无效位置
    df.loc[valid_positions.index, new_col] = labels
    label_positions = dict()
    for i in range(len(labels)):
        if labels[i] not in label_positions:
            label_positions[labels[i]] = []
            label_positions[labels[i]].append(coordinates[i])
        else:
            label_positions[labels[i]].append(coordinates[i])
    label_center = dict()
    for i in range(k+1):
        if i == 0:
            continue
        lon_list = [item[0] for item in label_positions[i]]
        lat_list = [item[1] for item in label_positions[i]]
        label_center[i] = [sum(lon_list)/len(lon_list), sum(lat_list)/len(lat_list)]
    cluster_center_col = "begin_cluster_center" if position_col == "begin_position" else "end_cluster_center"
    df[cluster_center_col] = [label_center[i] for i in labels]
    
    return df
  
  
def kmeans():
    df1 = pd.read_csv("/home/users/lichuan/shared/jld2csv/trips_150103.jld2_trips.csv")
    df2 = pd.read_csv("/home/users/lichuan/shared/jld2csv/trips_150104.jld2_trips.csv")
    df3 = pd.read_csv("/home/users/lichuan/shared/jld2csv/trips_150105.jld2_trips.csv")
    df4 = pd.read_csv("/home/users/lichuan/shared/jld2csv/trips_150106.jld2_trips.csv")
    df5 = pd.read_csv("/home/users/lichuan/shared/jld2csv/trips_150107.jld2_trips.csv")
    dataframes = [df1, df2, df3, df4, df5]
    df = pd.concat(dataframes, ignore_index=True)
    # 转换 tms 列
    df["lon"] = df["lon"].apply(safe_literal_eval)
    df["lat"] = df["lat"].apply(safe_literal_eval)
    df["begin_lon"] = df["lon"].apply(lambda x: x[0] if len(x) > 0 else None)
    df["begin_lat"] = df["lat"].apply(lambda x: x[0] if len(x) > 0 else None)
    df["end_lon"] = df["lon"].apply(lambda x: x[len(x)-1] if len(x) > 0 else None)
    df["end_lat"] = df["lat"].apply(lambda x: x[len(x)-1] if len(x) > 0 else None)
    df["begin_position"] = df.apply(
        lambda row: [row["lon"][0], row["lat"][0]] 
        if len(row["lon"]) > 0 and len(row["lat"]) > 0 
        else None,
        axis=1
    )

    # 生成 end_position 列（同理）
    df["end_position"] = df.apply(
        lambda row: [row["lon"][-1], row["lat"][-1]] 
        if len(row["lon"]) > 0 and len(row["lat"]) > 0 
        else None,
        axis=1
    )
    
    df = add_cluster_id(df, "begin_position", 5)
    df = add_cluster_id(df, "end_position", 5)
    print(df[['begin_position', 'begin_area_id', 'end_area_id', 'begin_cluster_center', 'end_cluster_center']].head(20))
    # df.to_csv('kmeans.csv')

kmeans()
