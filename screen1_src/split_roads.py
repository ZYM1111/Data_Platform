import ast # 请将此导入语句放在文件顶部

# lon_slice 和 lat_slice 参数当前未被此函数逻辑使用。
# 它们包含在内以匹配来自 api21_top_congested_roads 的调用签名。
def api13_split_trip_by_road(roads_data):
    """
    处理输入的 roads_data（应为路段 ID 列表或其字符串表示形式），
    并返回一个字典，其中键是唯一的 road_id，值是这些 road_id 在原始列表中的连续下标范围列表。
    每个范围表示为 (起始下标, 结束下标) 的元组。
    lon_slice 和 lat_slice 参数当前未被此函数逻辑使用。
    """
    roads_list = []
    if isinstance(roads_data, list):
        roads_list = roads_data
    elif isinstance(roads_data, str):
        try:
            # 使用 ast.literal_eval 以保证安全，替代 eval
            evaluated_data = ast.literal_eval(roads_data)
            # 确保解析结果是一个列表
            if isinstance(evaluated_data, list):
                roads_list = evaluated_data
            else:
                # print(f"警告：解析字符串 '{roads_data}' 未得到列表，实际为 {type(evaluated_data)}")
                return {} 
        except (ValueError, SyntaxError, TypeError):
            # 处理解析错误，例如字符串格式不正确
            # print(f"警告：无法将字符串 '{roads_data}' 解析为 Python 字面量。")
            return {}
    else:
        # print(f"警告：期望 roads_data 是列表或字符串，实际为 {type(roads_data)}")
        return {}

    if not roads_list: # 如果道路列表为空
        return {}

    # 步骤 1: 收集每个 road_id 出现的所有下标
    road_to_indices = {}
    for index, road_id in enumerate(roads_list):
        if road_id not in road_to_indices:
            road_to_indices[road_id] = []
        road_to_indices[road_id].append(index)

    # 步骤 2: 为每个 road_id 找到其下标的连续范围
    road_id_to_ranges = {}
    for road_id, indices in road_to_indices.items():
        # 'indices' 列表因为是通过 enumerate 获取的，所以已经是排序的
        if not indices: #理论上不会发生，因为 road_id 是从 road_to_indices 中获取的
            continue 
        
        current_road_ranges = []
        start_of_range = indices[0]
        
        for i in range(1, len(indices)):
            if indices[i] != indices[i-1] + 1:
                # 当前下标不连续，表示上一个范围结束
                current_road_ranges.append((start_of_range, indices[i-1]))
                # 开始新的范围
                start_of_range = indices[i]
        
        # 添加最后一个（或唯一的）范围
        current_road_ranges.append((start_of_range, indices[-1]))
        
        road_id_to_ranges[road_id] = current_road_ranges
        
    return road_id_to_ranges

if __name__ == "__main__":
    # # 示例调用
    # lon_slice = [1, 2, 2, 2, 3]
    # lat_slice = [4, 5, 2, 2, 6]
    roads_data = [1, 1, 2, 2, 3]
    
    result = api13_split_trip_by_road(roads_data)
    print(result)  # 输出: {1: [(0, 1)], 2: [(2, 3)], 3: [(4, 4)]}
