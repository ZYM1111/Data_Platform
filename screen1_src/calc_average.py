import pandas as pd
from ast import literal_eval
from math import radians, sin, cos, sqrt, atan2


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
    
    
def api14_average_driving_distance_and_time(df):
    # 读取 CSV 文件
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

if __name__ == "__main__":
    # 计算平均行驶距离和时间
    df = pd.read_csv("/home/users/lichuan/shared/jld2csv/trips_150107.jld2_trips.csv")
    average_distance, average_time = api14_average_driving_distance_and_time(df)
    print(f"Average Distance: {average_distance} km")
    print(f"Average Time: {average_time} minutes")
