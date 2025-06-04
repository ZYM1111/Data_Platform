from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
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
  


coords_1 = (39.9042, 116.4074)  # (纬度, 经度)
coords_2 = (31.2304, 121.4737)

print(haversine_distance(coords_1[0],coords_1[1],coords_2[0],coords_2[1]))