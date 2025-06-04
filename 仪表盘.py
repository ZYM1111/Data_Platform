import pandas as pd
from ast import literal_eval
from math import radians, sin, cos, sqrt, atan2
import datetime
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


def api21_top_congested_roads(time_begin, time_end, top_n=10):
    # 读取 CSV 文件
    df = pd.read_csv("./example.csv")

    # 转换 tms 列
    df["tms"] = df["tms"].apply(safe_literal_eval)
    df["begin_time"] = df["tms"].apply(lambda x: x[0] if len(x) > 0 else None)
    df["end_time"] = df["tms"].apply(lambda x: x[-1] if len(x) > 0 else None)

    df_filtered = df[(df["end_time"] >= time_begin) & (df["begin_time"] <= time_end)]

    road_count = {}
    for idx, row in df_filtered.iterrows():
        # print(row)
        begin_idx, end_idx = 0, len(row["tms"])
        
        # print(row["tms"][0])
        if time_begin > row["tms"][0]:
            begin_idx = next((i for i, x in enumerate(row["tms"]) if x > time_begin), -1)
        
        if time_end < row["tms"][-1]:
            end_idx = next((len(row["tms"]) - 1 - i for i, x in enumerate(reversed(row["tms"])) if x < time_end), -1)

        # if begin_idx == -1 or end_idx == -1:
            # print(row)
  
        # road_dict = {1: [1,1], 2: [2,2]}
        road_dict = api13_split_trip_by_road(row["lon"][begin_idx:end_idx], row["lat"][begin_idx:end_idx], row["roads"][begin_idx:end_idx]) 
        # road_dict = {1: [0,0], 2:[1,1]}

        for road_id in road_dict.keys():
            road_count[road_id] = road_count.get(road_id, 0) + 1

    # 按拥挤程度排序
    sorted_roads = sorted(road_count.items(), key=lambda x: x[1], reverse=True)
    top_road_ids = [road_id for road_id, _ in sorted_roads[:top_n]]

    return top_road_ids



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
  
  
# def kmeans():
    # df1 = pd.read_csv("/HDD/users/lichuan/shared/jld2csv/gcj_trips_150103.jld2_trips.csv")
    # df2 = pd.read_csv("/HDD/users/lichuan/shared/jld2csv/gcj_trips_150104.jld2_trips.csv")
    # df3 = pd.read_csv("/HDD/users/lichuan/shared/jld2csv/gcj_trips_150105.jld2_trips.csv")
    # df4 = pd.read_csv("/HDD/users/lichuan/shared/jld2csv/gcj_trips_150106.jld2_trips.csv")
    # df5 = pd.read_csv("/HDD/users/lichuan/shared/jld2csv/gcj_trips_150107.jld2_trips.csv")
    # dataframes = [df1, df2, df3, df4, df5]
    # df = pd.concat(dataframes, ignore_index=True)
    # # 转换 tms 列
    # df["lon"] = df["lon"].apply(safe_literal_eval)
    # df["lat"] = df["lat"].apply(safe_literal_eval)
    # df["begin_lon"] = df["lon"].apply(lambda x: x[0] if len(x) > 0 else None)
    # df["begin_lat"] = df["lat"].apply(lambda x: x[0] if len(x) > 0 else None)
    # df["end_lon"] = df["lon"].apply(lambda x: x[len(x)-1] if len(x) > 0 else None)
    # df["end_lat"] = df["lat"].apply(lambda x: x[len(x)-1] if len(x) > 0 else None)
    # df["begin_position"] = df.apply(
    #     lambda row: [row["lon"][0], row["lat"][0]] 
    #     if len(row["lon"]) > 0 and len(row["lat"]) > 0 
    #     else None,
    #     axis=1
    # )

    # # 生成 end_position 列（同理）
    # df["end_position"] = df.apply(
    #     lambda row: [row["lon"][-1], row["lat"][-1]] 
    #     if len(row["lon"]) > 0 and len(row["lat"]) > 0 
    #     else None,
    #     axis=1
    # )
    
    # df = add_cluster_id(df, "begin_position", 5)
    # df = add_cluster_id(df, "end_position", 5)
    # print(df[['begin_position', 'begin_area_id', 'end_area_id', 'begin_cluster_center', 'end_cluster_center']].head(20))
    # df.to_csv('kmeans.csv')

from tqdm import tqdm
def fig31():

    df1 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150103.jld2_trips.csv")
    df2 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150104.jld2_trips.csv")
    df3 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150105.jld2_trips.csv")
    df4 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150106.jld2_trips.csv")
    df5 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150107.jld2_trips.csv")

    #取df1随机100条数据
    df1 = df1.sample(n=100, random_state=42)
    df2 = df2.sample(n=100, random_state=42)
    df3 = df3.sample(n=100, random_state=42)
    df4 = df4.sample(n=100, random_state=42)
    df5 = df5.sample(n=100, random_state=42)


    df1["date"] = "20150103"
    df2["date"] = "20150104"
    df3["date"] = "20150105"
    df4["date"] = "20150106"
    df5["date"] = "20150107"

    # 拼接所有 DataFrame，向下拼接
    # print(len(df1))
    df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    # df = df2
    #创建id列
    # df["id"] = df.index + 1
    # print(len(df))
    df["tms"] = df["tms"].apply(safe_literal_eval)
    df["lon"] = df["lon"].apply(safe_literal_eval)
    df["lat"] = df["lat"].apply(safe_literal_eval)

    ddf = pd.DataFrame()
    
    ids = 0
    #使用tqdm显示进度条
    for index, row in tqdm(df.iterrows(), total=len(df)):
    # for index, row in df.iterrows():
        lon, lat, tms = row["lon"], row["lat"], row["tms"]
        date = row["date"]
        for i in range(len(lon)-1):
            #ddf['lon']的第i个元素是lon[i]
            ddf.loc[ids, 'begin_lon'] = lon[i]
            ddf.loc[ids, 'begin_lat'] = lat[i]
            ddf.loc[ids, 'end_lon'] = lon[i+1]
            ddf.loc[ids, 'end_lat'] = lat[i+1]
            # ddf.loc[ids, 'tms'] = tms[i]
            ddf.loc[ids, 'date'] = date
            ids += 1
    #将df[begin_lon, begin_lat]保存到csv文件中
    # features = ["lon", "lat", "tms", "begin_lon""b、egin_lat", "end_lon", "end_lat", "date"]
    # 保存df前100条数据到csv文件中
    ddf.head(100).to_csv("./fig31_eamples_.csv", index=False)
    ddf.to_csv("./fig31_.csv", index=False)
    # df.to_csv("./fig3.csv", index=False)

def fig32():
    df1 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150103.jld2_trips.csv")
    df2 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150104.jld2_trips.csv")
    df3 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150105.jld2_trips.csv")
    df4 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150106.jld2_trips.csv")
    df5 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150107.jld2_trips.csv")

    df = pd.DataFrame()
    df["date"] = ["20150103", "20150104", "20150105", "20150106", "20150107"]
    df["count"] = [len(df1), len(df2), len(df3), len(df4), len(df5)]
    # 将20150103提取星期几
    df["weekday"] = [datetime.datetime.strptime(date, "%Y%m%d").weekday() for date in df["date"]]

    df.to_csv("./fig32.csv", index=False)


# import datetime

def get_time_period(hour):
    if 0 <= hour < 6:
        return "凌晨时段（0:00-6:00）"
    elif 6 <= hour < 9:
        return "早高峰时段（6:00-9:00）"
    elif 9 <= hour < 16:
        return "白天平峰时段（9:00-16:00）"
    elif 16 <= hour < 19:
        return "晚高峰时段（16:00-19:00）"
    elif 19 <= hour < 24:
        return "夜间时段（19:00-24:00）"
    else:
        return "未知时段"

def classify_trajectory(ts_list):
    period_count = {}
    for ts in ts_list:
        hour = datetime.datetime.fromtimestamp(ts).hour
        period = get_time_period(hour)
        period_count[period] = period_count.get(period, 0) + 1
    
    # print(period_count)
    main_period = max(period_count, key=period_count.get)
    return main_period

def fig33():
    df1 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150103.jld2_trips.csv")
    df2 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150104.jld2_trips.csv")
    df3 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150105.jld2_trips.csv")
    df4 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150106.jld2_trips.csv")
    df5 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150107.jld2_trips.csv")

    df1["date"] = "20150103"
    df2["date"] = "20150104"
    df3["date"] = "20150105"
    df4["date"] = "20150106"
    df5["date"] = "20150107"

    df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    df["id"] = df.index + 1

    df["tms"] = df["tms"].apply(safe_literal_eval)
    for index, row in df.iterrows():
        df.loc[index, 'time_period'] = classify_trajectory(row["tms"])
    
    features = ["id", "date", "time_period"]
    # time_period值是否有不是早高峰时段（6:00-9:00）的
    # print(df['time_period'].value_counts())
    df[features].head(100).to_csv("./fig33_eamples_.csv", index=False)
    df[features].to_csv("./fig33.csv", index=False)


def fig41_recommend():
    df1 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150103.jld2_trips.csv")
    df2 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150104.jld2_trips.csv")
    df3 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150105.jld2_trips.csv")
    df4 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150106.jld2_trips.csv")
    df5 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150107.jld2_trips.csv")
    df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    df["id"] = df.index + 1

    df["lon"] = df["lon"].apply(safe_literal_eval)
    df["lat"] = df["lat"].apply(safe_literal_eval)

    df["begin_lon"] = df["lon"].apply(lambda x: x[0] if len(x) > 0 else None)
    df["begin_lat"] = df["lat"].apply(lambda x: x[0] if len(x) > 0 else None)
    
    df["driving_distance_km"] = df.apply(
    lambda row: calculate_driving_distance(row["lon"], row["lat"]), 
    axis=1)

    # 如果driving_distance_km大于15km，df['flag']=long_trip,如果小于5km,df['flag']=short_trip,否则df['flag']=normal_trip
    df['flag'] = df['driving_distance_km'].apply(lambda x: 'long_trip' if x > 30 else 'short_trip' if x < 2 else 'normal_trip')

    # 去除flag==normal_trip的行
    df = df[df['flag'] != 'normal_trip']
    df1 = df[df['flag'] == 'long_trip']
    df1["begin_position"] = df1.apply(
        lambda row: [row["lon"][0], row["lat"][0]] 
        if len(row["lon"]) > 0 and len(row["lat"]) > 0 
        else None,
        axis=1
    )
    df1 = add_cluster_id(df1, "begin_position", 5)

    df2 = df[df['flag'] == 'short_trip']
    df2["begin_position"] = df2.apply(
        lambda row: [row["lon"][0], row["lat"][0]]
        if len(row["lon"]) > 0 and len(row["lat"]) > 0
        else None,
        axis=1
    )
    df2 = add_cluster_id(df2, "begin_position", 5)
    df = pd.concat([df1, df2], ignore_index=True)

    # 保存df前100条数据到csv文件中
    features = ["id", "flag", "begin_position", "begin_area_id", "begin_cluster_center"]
    df[features].head(20).to_csv("./fig41_eamples.csv", index=False)
    df[features].to_csv("./fig41.csv", index=False)
    # features = ["id", "flag", ""]

def fig42():
    # /HDD/users/lichuan/shared/gcj_trips_150103.jld2_trips.csv
    df1 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150103.jld2_trips.csv")
    df2 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150104.jld2_trips.csv")
    df3 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150105.jld2_trips.csv")
    df4 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150106.jld2_trips.csv")
    df5 = pd.read_csv("/HDD/users/lichuan/shared/gcj_trips_150107.jld2_trips.csv")
    
    df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    df["id"] = df.index + 1

    df["lon"] = df["lon"].apply(safe_literal_eval)
    df["lat"] = df["lat"].apply(safe_literal_eval)
    df['begin_lon'] = df['lon'].apply(lambda x: x[0] if len(x) > 0 else None)
    df['begin_lat'] = df['lat'].apply(lambda x: x[0] if len(x) > 0 else None)

    df['end_lon'] = df['lon'].apply(lambda x: x[-1] if len(x) > 0 else None)
    df['end_lat'] = df['lat'].apply(lambda x: x[-1] if len(x) > 0 else None)

    points = []
    for index, row in df.iterrows():
        begin_position = (row['begin_lon'], row['begin_lat'])
        end_position = (row['end_lon'], row['end_lat'])
        points.append(begin_position)
        points.append(end_position)

    points = np.array(points) 
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(points)
    business_areas = []
    for i in range(n_clusters):
        cluster_points = points[labels == i]
        left_lon = cluster_points[:, 0].min()
        right_lon = cluster_points[:, 0].max()
        top_lat = cluster_points[:, 1].max()
        bottom_lat = cluster_points[:, 1].min()
        mean_lon = cluster_points[:, 0].mean()
        mean_lat = cluster_points[:, 1].mean()
        print(f'({mean_lon}, {mean_lat})')
        business_areas.append([left_lon, right_lon, top_lat, bottom_lat])
    print(business_areas)

    # 保存df前100条数据到csv文件中
    results = {}
    for index, row in df.iterrows():
        begin_position = (row['begin_lon'], row['begin_lat'])
        end_position = (row['end_lon'], row['end_lat'])
        for i, area in enumerate(business_areas):
            if i not in results:
                results[i] = {}
            left_lon, right_lon, top_lat, bottom_lat = area
            if left_lon <= begin_position[0] <= right_lon and bottom_lat <= begin_position[1] <= top_lat:
                if row['devid'] not in results[i]:
                    results[i][row['devid']] = 0
                results[i][row['devid']] += 1
            if left_lon <= end_position[0] <= right_lon and bottom_lat <= end_position[1] <= top_lat:
                if row['devid'] not in results[i]:
                    results[i][row['devid']] = 0
                results[i][row['devid']] += 1
    
    #  对results[i]的values排序从大到小
    for key, values in results.items():
        results[key] = dict(sorted(values.items(), key=lambda item: item[1], reverse=True))
        # 取二十个
        results[key] = dict(list(results[key].items())[:20])
    
    #怎么将results保存到dataframe中
    shops, devids, counts = [], [], []
    for key, values in results.items():
        for devid, count in values.items():
            shops.append(key)
            devids.append(devid)
            counts.append(count)
            
    df = pd.DataFrame({'shop': shops, 'devid': devids, 'count': counts})
    df.to_csv('./fig42.csv', index=False)
fig41_recommend()    

