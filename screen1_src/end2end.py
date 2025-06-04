

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


def add_cluster_id(df, position_col='begin_position', k=20):
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
  
def calculate_od_flow(df, top_n=10):
    """计算聚类间通信量并返回top_n"""
    # 统计每个OD对的行程数量
    od_flows = []
    
    # 遍历所有可能的OD组合
    for pickup_cluster in range(1, top_n + 1):  # 修改这里：使用top_n而不是硬编码的6
        for dropoff_cluster in range(1, top_n + 1):  # 修改这里：使用top_n而不是硬编码的6
            # 统计从pickup_cluster到dropoff_cluster的行程数
            flow_count = len(df[(df['begin_area_id'] == pickup_cluster) & 
                               (df['end_area_id'] == dropoff_cluster)])
            
            if flow_count > 0:  # 只记录有流量的OD对
                od_flows.append({
                    'pickup_cluster': pickup_cluster,
                    'dropoff_cluster': dropoff_cluster,
                    'flow_count': flow_count
                })
    
    # 按流量降序排序，取top_n
    od_flows.sort(key=lambda x: x['flow_count'], reverse=True)
    return od_flows[:top_n]

def generate_cluster_communication_csv(df, top_n=10):
    """生成聚类中心通信量CSV，只包含lon、lat、通信量三列"""
    cluster_stats = []
    
    # 统计每个聚类作为起点和终点的总通信量
    for cluster_id in range(1, top_n + 1):  # 修改这里：使用top_n而不是硬编码的6
        # 作为起点的通信量
        as_origin = len(df[df['begin_area_id'] == cluster_id])
        # 作为终点的通信量  
        as_destination = len(df[df['end_area_id'] == cluster_id])
        # 总通信量
        total_flow = as_origin + as_destination
        
        # 获取聚类中心坐标（从起点聚类中心获取）
        cluster_data = df[df['begin_area_id'] == cluster_id]
        if not cluster_data.empty:
            center = cluster_data['begin_cluster_center'].iloc[0]
            lon, lat = center[0], center[1]
        else:
            # 如果作为起点没有数据，从终点获取
            cluster_data = df[df['end_area_id'] == cluster_id]
            if not cluster_data.empty:
                center = cluster_data['end_cluster_center'].iloc[0]
                lon, lat = center[0], center[1]
            else:
                lon, lat = 0, 0
        
        cluster_stats.append({
            'lon': round(lon, 6),
            'lat': round(lat, 6),
            'communication_flow': total_flow
        })
    
    # 按通信量降序排序，取top_n
    cluster_stats.sort(key=lambda x: x['communication_flow'], reverse=True)
    return cluster_stats  # 这里不需要再切片，因为上面已经循环了top_n个
  
def generate_od_flow_csv(df, top_n=10):
    """生成OD流量CSV，包含上车点经纬度、下车点经纬度和通信量五列"""
    od_flows = []
    
    # 获取所有聚类中心坐标
    cluster_centers = {}
    
    # 获取上车点聚类中心
    for cluster_id in range(1, top_n + 1):
        cluster_data = df[df['begin_area_id'] == cluster_id]
        if not cluster_data.empty:
            center = cluster_data['begin_cluster_center'].iloc[0]
            cluster_centers[f'begin_{cluster_id}'] = [center[0], center[1]]
    
    # 获取下车点聚类中心
    for cluster_id in range(1, top_n + 1):
        cluster_data = df[df['end_area_id'] == cluster_id]
        if not cluster_data.empty:
            center = cluster_data['end_cluster_center'].iloc[0]
            cluster_centers[f'end_{cluster_id}'] = [center[0], center[1]]
    
    # 遍历所有可能的OD组合
    for pickup_cluster in range(1, top_n + 1):
        for dropoff_cluster in range(1, top_n + 1):
            # 统计从pickup_cluster到dropoff_cluster的行程数
            flow_count = len(df[(df['begin_area_id'] == pickup_cluster) & 
                               (df['end_area_id'] == dropoff_cluster)])
            
            # 获取上车点和下车点坐标
            pickup_key = f'begin_{pickup_cluster}'
            dropoff_key = f'end_{dropoff_cluster}'
            
            # 如果坐标存在，添加到结果中（即使流量为0也添加）
            if pickup_key in cluster_centers and dropoff_key in cluster_centers:
                pickup_lon, pickup_lat = cluster_centers[pickup_key]
                dropoff_lon, dropoff_lat = cluster_centers[dropoff_key]
                
                od_flows.append({
                    'pickup_lon': round(pickup_lon, 6),
                    'pickup_lat': round(pickup_lat, 6),
                    'dropoff_lon': round(dropoff_lon, 6),
                    'dropoff_lat': round(dropoff_lat, 6),
                    'flow_count': flow_count
                })
    
    return od_flows

def kmeans():
    df1 = pd.read_csv("/home/users/lichuan/shared/gcj_trips_150103.jld2_trips.csv")
    df2 = pd.read_csv("/home/users/lichuan/shared/gcj_trips_150104.jld2_trips.csv")
    df3 = pd.read_csv("/home/users/lichuan/shared/gcj_trips_150105.jld2_trips.csv")
    df4 = pd.read_csv("/home/users/lichuan/shared/gcj_trips_150106.jld2_trips.csv")
    df5 = pd.read_csv("/home/users/lichuan/shared/gcj_trips_150107.jld2_trips.csv")
    dataframes = [df1, df2, df3, df4, df5]
    df = pd.concat(dataframes, ignore_index=True)
    
    # 转换数据
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
    df["end_position"] = df.apply(
        lambda row: [row["lon"][-1], row["lat"][-1]] 
        if len(row["lon"]) > 0 and len(row["lat"]) > 0 
        else None,
        axis=1
    )
    
    top_n = 10  # 修改为10个聚类

    # 聚类分析
    df = add_cluster_id(df, "begin_position", top_n)
    df = add_cluster_id(df, "end_position", top_n)
    
    # 生成OD流量数据
    od_flows = generate_od_flow_csv(df, top_n=top_n)
    
    # 打印结果统计
    print("=== OD流量统计 ===")
    print(f"总共生成 {len(od_flows)} 条OD记录")
    print(f"有流量的OD对: {len([od for od in od_flows if od['flow_count'] > 0])}")
    print(f"无流量的OD对: {len([od for od in od_flows if od['flow_count'] == 0])}")
    
    # 显示前10条记录
    print("\n前10条OD流量记录:")
    for i, od in enumerate(od_flows[:10], 1):
        print(f"{i}. 上车({od['pickup_lon']}, {od['pickup_lat']}) -> "
              f"下车({od['dropoff_lon']}, {od['dropoff_lat']}): {od['flow_count']}次")
    
    # 保存到CSV
    import os
    os.makedirs('screen1', exist_ok=True)
    
    # 保存OD流量数据
    od_df = pd.DataFrame(od_flows)
    od_df.to_csv('screen1/od_flow_matrix.csv', index=False)
    
    print(f"\n总行程数: {len(df)}")
    print("OD流量矩阵已保存到 screen1/od_flow_matrix.csv")
    print(f"CSV文件包含 {len(od_df)} 行，5列数据")


kmeans()
