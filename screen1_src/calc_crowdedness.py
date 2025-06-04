import pandas as pd
from ast import literal_eval
from split_roads import api13_split_trip_by_road
import math
from datetime import datetime, timezone
import os
from pathlib import Path


# =============================================================================
# 工具函数 (Utility Functions)
# =============================================================================

def safe_literal_eval(s):
    """安全地解析字符串为Python对象(列表)"""
    if isinstance(s, str):
        if s.startswith("Any[") and s.endswith("]"):
            list_str = s[4:-1]
            try:
                return literal_eval(f"[{list_str}]")
            except:
                return []
        else:
            try:
                return literal_eval(s.replace(';', ','))
            except:
                return []
    elif isinstance(s, list):
        return s
    return []


# =============================================================================
# 时间转换函数 (Time Conversion Functions)
# =============================================================================

def unix_timestamp_to_datetime_str(unix_ts):
    """将 Unix 时间戳转换为 'YYYY-MM-DD HH:MM:SS' 格式的字符串 (UTC)"""
    if unix_ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(unix_ts), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    except (ValueError, TypeError):
        return "Invalid Timestamp"


def datetime_str_to_unix_timestamp(dt_str):
    """将 'YYYY-MM-DD HH:MM:SS' 格式的字符串转换为 Unix 时间戳 (假设输入为UTC时间)"""
    if dt_str is None:
        return None
    try:
        dt_obj = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        # 明确指定为UTC时区，因为数据集中的时间戳是UTC的
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        return dt_obj.timestamp()
    except (ValueError, TypeError):
        return "Invalid Datetime String"


# =============================================================================
# 数据预处理函数 (Data Preprocessing Functions)
# =============================================================================

def preprocess_dataframe(df):
    """预处理DataFrame，转换tms和roads列为列表格式"""
    df_processed = df.copy()
    df_processed["tms"] = df_processed["tms"].apply(safe_literal_eval)
    df_processed["roads"] = df_processed["roads"].apply(safe_literal_eval)
    return df_processed


def load_and_preprocess_csv(csv_file_path):
    """加载并预处理CSV文件"""
    try:
        df_main = pd.read_csv(csv_file_path)
        df_main = preprocess_dataframe(df_main)
        print(f"成功从 {csv_file_path} 加载并预处理数据。")
        return df_main
    except FileNotFoundError:
        print(f"错误: CSV文件未找到 {csv_file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"加载或预处理CSV时发生错误: {e}")
        return pd.DataFrame()


def load_multiple_days_data(base_path, date_list, file_prefix="gcj_trips"):
    """加载多天的数据"""
    all_data = []
    for date_str in date_list:
        file_path = f"{base_path}/{file_prefix}_{date_str}.jld2_trips.csv"
        df = load_and_preprocess_csv(file_path)
        if not df.empty:
            df['date'] = date_str
            all_data.append(df)
        else:
            print(f"警告: 日期 {date_str} 的数据加载失败")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


# =============================================================================
# 核心分析函数 (Core Analysis Functions)
# =============================================================================

def api21_top_congested_roads(df_input, time_begin_str, time_end_str, top_n=10):
    """
    分析DataFrame中的行程数据，找出在指定时间范围内的最拥挤路段。
    
    参数:
        df_input: 预加载并已处理 'tms' 和 'roads' 列的DataFrame
        time_begin_str: 'YYYY-MM-DD HH:MM:SS' 格式的开始时间字符串
        time_end_str: 'YYYY-MM-DD HH:MM:SS' 格式的结束时间字符串，或 None
        top_n: 返回最拥挤路段的数量
    
    返回:
        list: [(road_id, count), ...] 按拥堵程度排序的路段列表
    """
    df = df_input.copy()

    # 时间转换和验证
    time_begin_unix = datetime_str_to_unix_timestamp(time_begin_str)
    if not isinstance(time_begin_unix, (int, float)):
        raise ValueError(f"起始时间格式无效或转换失败: {time_begin_str}")

    if time_end_str:
        time_end_unix = datetime_str_to_unix_timestamp(time_end_str)
        if not isinstance(time_end_unix, (int, float)):
            raise ValueError(f"结束时间格式无效或转换失败: {time_end_str}")
    else:
        time_end_unix = float('inf')

    if time_begin_unix >= time_end_unix:
        raise ValueError("开始时间必须早于结束时间。")

    # 提取行程的开始和结束时间
    df["begin_time"] = df["tms"].apply(lambda x: x[0] if x and len(x) > 0 and isinstance(x[0], (int, float)) else None)
    df["end_time"] = df["tms"].apply(lambda x: x[-1] if x and len(x) > 0 and isinstance(x[-1], (int, float)) else None)

    # 筛选有效数据
    df.dropna(subset=['begin_time', 'end_time'], inplace=True)
    df_filtered = df[(df["end_time"] >= time_begin_unix) & (df["begin_time"] <= time_end_unix)]

    # 统计路段使用频次
    road_count = {}
    for idx, row in df_filtered.iterrows():
        road_segment_to_analyze = _extract_road_segments_in_timerange(
            row["tms"], row["roads"], time_begin_unix, time_end_unix
        )
        
        road_dict = api13_split_trip_by_road(road_segment_to_analyze)
        for road_id in road_dict.keys():
            road_count[road_id] = road_count.get(road_id, 0) + 1

    return sorted(road_count.items(), key=lambda x: x[1], reverse=True)[:top_n]


def _extract_road_segments_in_timerange(tms_list, roads_list, time_begin_unix, time_end_unix):
    """提取指定时间范围内的路段"""
    if not (tms_list and roads_list and len(tms_list) == len(roads_list) and 
            all(isinstance(ts, (int, float)) for ts in tms_list)):
        return []

    start_slice_idx = 0
    while start_slice_idx < len(tms_list) and tms_list[start_slice_idx] < time_begin_unix:
        start_slice_idx += 1

    end_slice_idx = 0
    while end_slice_idx < len(tms_list) and tms_list[end_slice_idx] <= time_end_unix:
        end_slice_idx += 1

    if start_slice_idx < end_slice_idx:
        return roads_list[start_slice_idx:end_slice_idx]
    return []


def count_trips_per_hour(df_input):
    """
    统计DataFrame中每个小时时段的轨迹数量。
    
    参数:
        df_input: 已预处理的DataFrame，'tms'列为列表格式
    
    返回:
        dict: {hour: count} 每小时的轨迹数量
    """
    hourly_counts = {hour: 0 for hour in range(24)}

    if df_input.empty:
        print("输入DataFrame为空，无法统计每小时轨迹数。")
        return hourly_counts

    for index, row in df_input.iterrows():
        tms_list = row['tms']
        if tms_list and isinstance(tms_list, list) and len(tms_list) > 0:
            first_timestamp = tms_list[0]
            if isinstance(first_timestamp, (int, float)):
                try:
                    dt_object_utc = datetime.fromtimestamp(float(first_timestamp), tz=timezone.utc)
                    hour_of_day = dt_object_utc.hour
                    hourly_counts[hour_of_day] += 1
                except (ValueError, TypeError, OSError) as e:
                    print(f"警告: 行 {index} 的时间戳 {first_timestamp} 无效或无法转换: {e}")
            else:
                print(f"警告: 行 {index} 的第一个时间戳不是有效数字: {first_timestamp}")

    return hourly_counts


# =============================================================================
# 统计分析函数 
# =============================================================================

def generate_single_day_hourly_stats(df_input, output_path, date_str):
    """生成单天24小时打车订单统计CSV"""
    hourly_counts = count_trips_per_hour(df_input)
    
    # 准备数据
    data = []
    for hour in range(24):
        data.append({
            'hour': hour,
            'time_range': f"{hour:02d}:00-{hour:02d}:59",
            'trip_count': hourly_counts[hour],
            'date': date_str
        })
    
    # 保存CSV
    df_result = pd.DataFrame(data)
    csv_filename = f"hourly_trips_{date_str}.csv"
    full_path = os.path.join(output_path, csv_filename)
    df_result.to_csv(full_path, index=False, encoding='utf-8')
    print(f"单天小时统计已保存到: {full_path}")
    return df_result


def generate_day_comparison_stats(df_today, df_yesterday, output_path, today_date, yesterday_date):
    """生成今日相对昨日同期增长统计CSV"""
    today_hourly = count_trips_per_hour(df_today)
    yesterday_hourly = count_trips_per_hour(df_yesterday)
    
    # 准备数据
    data = []
    for hour in range(24):
        today_count = today_hourly[hour]
        yesterday_count = yesterday_hourly[hour]
        
        if yesterday_count > 0:
            growth_rate = ((today_count - yesterday_count) / yesterday_count) * 100
        else:
            growth_rate = float('inf') if today_count > 0 else 0
        
        growth_absolute = today_count - yesterday_count
        
        data.append({
            'hour': hour,
            'time_range': f"{hour:02d}:00-{hour:02d}:59",
            'today_count': today_count,
            'yesterday_count': yesterday_count,
            'growth_absolute': growth_absolute,
            'growth_rate_percent': growth_rate,
            'today_date': today_date,
            'yesterday_date': yesterday_date
        })
    
    # 保存CSV
    df_result = pd.DataFrame(data)
    csv_filename = f"growth_comparison_{today_date}_vs_{yesterday_date}.csv"
    full_path = os.path.join(output_path, csv_filename)
    df_result.to_csv(full_path, index=False, encoding='utf-8')
    print(f"日期对比统计已保存到: {full_path}")
    return df_result


def generate_multi_day_aggregated_stats(df_multi_day, output_path, date_list):
    """生成多天汇总的24小时统计CSV"""
    # 按小时汇总所有天的数据
    aggregated_hourly = {hour: 0 for hour in range(24)}
    
    for index, row in df_multi_day.iterrows():
        tms_list = row['tms']
        if tms_list and isinstance(tms_list, list) and len(tms_list) > 0:
            first_timestamp = tms_list[0]
            if isinstance(first_timestamp, (int, float)):
                try:
                    dt_object_utc = datetime.fromtimestamp(float(first_timestamp), tz=timezone.utc)
                    hour_of_day = dt_object_utc.hour
                    aggregated_hourly[hour_of_day] += 1
                except (ValueError, TypeError, OSError):
                    pass
    
    # 准备数据
    data = []
    for hour in range(24):
        data.append({
            'hour': hour,
            'time_range': f"{hour:02d}:00-{hour:02d}:59",
            'total_trip_count': aggregated_hourly[hour],
            'average_per_day': aggregated_hourly[hour] / len(date_list),
            'date_range': f"{min(date_list)}_to_{max(date_list)}",
            'days_included': len(date_list)
        })
    
    # 保存CSV
    df_result = pd.DataFrame(data)
    date_range_str = f"{min(date_list)}_to_{max(date_list)}"
    csv_filename = f"aggregated_hourly_trips_{date_range_str}.csv"
    full_path = os.path.join(output_path, csv_filename)
    df_result.to_csv(full_path, index=False, encoding='utf-8')
    print(f"多天汇总统计已保存到: {full_path}")
    return df_result


def generate_period_comparison_stats(df_recent, df_previous, output_path, recent_dates, previous_dates):
    """生成两个时期的增长对比统计CSV"""
    # 计算两个时期的小时汇总
    recent_hourly = {hour: 0 for hour in range(24)}
    previous_hourly = {hour: 0 for hour in range(24)}
    
    # 统计最近时期
    for index, row in df_recent.iterrows():
        tms_list = row['tms']
        if tms_list and isinstance(tms_list, list) and len(tms_list) > 0:
            first_timestamp = tms_list[0]
            if isinstance(first_timestamp, (int, float)):
                try:
                    dt_object_utc = datetime.fromtimestamp(float(first_timestamp), tz=timezone.utc)
                    hour_of_day = dt_object_utc.hour
                    recent_hourly[hour_of_day] += 1
                except (ValueError, TypeError, OSError):
                    pass
    
    # 统计之前时期
    for index, row in df_previous.iterrows():
        tms_list = row['tms']
        if tms_list and isinstance(tms_list, list) and len(tms_list) > 0:
            first_timestamp = tms_list[0]
            if isinstance(first_timestamp, (int, float)):
                try:
                    dt_object_utc = datetime.fromtimestamp(float(first_timestamp), tz=timezone.utc)
                    hour_of_day = dt_object_utc.hour
                    previous_hourly[hour_of_day] += 1
                except (ValueError, TypeError, OSError):
                    pass
    
    # 准备数据
    data = []
    for hour in range(24):
        recent_count = recent_hourly[hour]
        previous_count = previous_hourly[hour]
        
        if previous_count > 0:
            growth_rate = ((recent_count - previous_count) / previous_count) * 100
        else:
            growth_rate = float('inf') if recent_count > 0 else 0
        
        growth_absolute = recent_count - previous_count
        
        data.append({
            'hour': hour,
            'time_range': f"{hour:02d}:00-{hour:02d}:59",
            'recent_period_count': recent_count,
            'previous_period_count': previous_count,
            'growth_absolute': growth_absolute,
            'growth_rate_percent': growth_rate,
            'recent_period_avg_per_day': recent_count / len(recent_dates),
            'previous_period_avg_per_day': previous_count / len(previous_dates),
            'recent_dates': "_".join(recent_dates),
            'previous_dates': "_".join(previous_dates)
        })
    
    # 保存CSV
    df_result = pd.DataFrame(data)
    recent_range = f"{min(recent_dates)}_to_{max(recent_dates)}"
    previous_range = f"{min(previous_dates)}_to_{max(previous_dates)}"
    csv_filename = f"period_comparison_{recent_range}_vs_{previous_range}.csv"
    full_path = os.path.join(output_path, csv_filename)
    df_result.to_csv(full_path, index=False, encoding='utf-8')
    print(f"时期对比统计已保存到: {full_path}")
    return df_result


def generate_congestion_analysis_csv(df_input, time_begin_str, time_end_str, output_path, analysis_name, top_n=50):
    """生成道路拥堵分析CSV"""
    try:
        congested_roads = api21_top_congested_roads(df_input, time_begin_str, time_end_str, top_n)
        
        # 准备数据
        data = []
        for rank, (road_id, count) in enumerate(congested_roads, 1):
            data.append({
                'rank': rank,
                'road_id': road_id,
                'congestion_count': count,
                'time_begin': time_begin_str,
                'time_end': time_end_str,
                'analysis_name': analysis_name
            })
        
        # 保存CSV
        df_result = pd.DataFrame(data)
        csv_filename = f"congestion_analysis_{analysis_name}.csv"
        full_path = os.path.join(output_path, csv_filename)
        df_result.to_csv(full_path, index=False, encoding='utf-8')
        print(f"拥堵分析已保存到: {full_path}")
        return df_result
    
    except Exception as e:
        print(f"生成拥堵分析CSV时出错: {e}")
        return pd.DataFrame()


# =============================================================================
# 测试和分析函数 (Testing and Analysis Functions)
# =============================================================================

def test_time_conversions(df_sample=None):
    """测试时间转换函数并显示CSV中的样本转换"""
    print("--- 手动时间转换测试 ---")
    sample_unix_ts = 1420243200.0
    print(f"Unix {sample_unix_ts} -> Datetime: {unix_timestamp_to_datetime_str(sample_unix_ts)}")

    sample_dt_str_utc = "2015-01-03 00:03:05"
    print(f"Datetime '{sample_dt_str_utc}' -> Unix: {datetime_str_to_unix_timestamp(sample_dt_str_utc)}")
    print("--------------------------\n")

    if df_sample is not None and not df_sample.empty:
        print("--- CSV样本时间转换验证 ---")
        temp_tms_series = df_sample["tms"].apply(safe_literal_eval)
        count = 0
        for tms_list in temp_tms_series:
            if tms_list and len(tms_list) > 0:
                original_unix_ts = tms_list[0]
                converted_dt_str = unix_timestamp_to_datetime_str(original_unix_ts)
                print(f"原始Unix时间戳: {original_unix_ts} -> 转换后: {converted_dt_str}")
                count += 1
                if count >= 5:
                    break
        if count == 0:
            print("在提供的样本中未找到有效的 'tms' 数据进行转换。")
        print("------------------------------------\n")


def analyze_dataset_timerange(df):
    """分析数据集的时间范围"""
    print("--- 数据集时间范围检查 ---")
    all_timestamps = []
    for tms_list in df["tms"]:
        if isinstance(tms_list, list):
            for ts in tms_list:
                if isinstance(ts, (int, float)):
                    all_timestamps.append(float(ts))

    if all_timestamps:
        min_ts = min(all_timestamps)
        max_ts = max(all_timestamps)
        print(f"数据集中最早的时间戳: {min_ts} -> {unix_timestamp_to_datetime_str(min_ts)}")
        print(f"数据集中最晚的时间戳: {max_ts} -> {unix_timestamp_to_datetime_str(max_ts)}")
    else:
        print("在数据集中未找到有效的时间戳来确定范围。")
    print("---------------------------\n")


def print_hourly_statistics(trip_counts_by_hour):
    """打印每小时轨迹数量统计"""
    print("--- 每小时轨迹数量统计 ---")
    for hour, count in sorted(trip_counts_by_hour.items()):
        print(f"小时 {hour:02d}:00 - {hour:02d}:59 : {count} 条轨迹")
    print("----------------------------\n")


def run_congestion_analysis(df, time_begin_str, time_end_str, top_n=10):
    """运行拥堵分析"""
    print(f"--- 调用 api21_top_congested_roads ---")
    print(f"查询时间范围: {time_begin_str} 到 {time_end_str}")
    
    api_call_begin_unix = datetime_str_to_unix_timestamp(time_begin_str)
    api_call_end_unix = datetime_str_to_unix_timestamp(time_end_str)
    print(f"API 调用将使用 Unix 开始时间: {api_call_begin_unix}")
    print(f"API 调用将使用 Unix 结束时间: {api_call_end_unix}")
    print("-------------------------------------\n")

    try:
        congested_roads_details = api21_top_congested_roads(df, time_begin_str, time_end_str, top_n)
        print(f"Top {top_n} Congested Roads and their counts:")
        if congested_roads_details:
            for road_id, count in congested_roads_details:
                print(f"Road ID: {road_id}, Congestion Count: {count}")
        else:
            print("在指定时间范围内未找到拥堵路段。")
    except ValueError as e:
        print(f"API调用错误: {e}")
    except Exception as e:
        print(f"API调用时发生意外错误: {e}")


# =============================================================================
# 主程序 (Main Program)
# =============================================================================

def main():
    """主函数"""
    BASE_PATH = "/home/users/lichuan/shared"
    OUTPUT_PATH = "./screen1"
    
    # 创建输出目录
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # 定义分析日期
    TODAY = "150107"        # 1月7日
    YESTERDAY = "150106"    # 1月6日  
    RECENT_3_DAYS = ["150107", "150106", "150105"]  # 最近三天
    PREVIOUS_3_DAYS = ["150105", "150104", "150103"]  # 之前三天
    ALL_DATES = ["150107", "150106", "150105", "150104", "150103"]  # 所有需要的日期
    
    print("=== 开始综合分析 ===\n")
    
    # 一次性加载所有需要的数据
    print("加载所有数据...")
    df_all = load_multiple_days_data(BASE_PATH, ALL_DATES, "gcj_trips")
    
    if df_all.empty:
        print("错误: 无法加载任何数据")
        return
    
    # 从完整数据集中筛选出不同日期的数据
    df_today = df_all[df_all['date'] == TODAY].copy()
    df_yesterday = df_all[df_all['date'] == YESTERDAY].copy()
    df_recent_3 = df_all[df_all['date'].isin(RECENT_3_DAYS)].copy()
    df_previous_3 = df_all[df_all['date'].isin(PREVIOUS_3_DAYS)].copy()
    
    # 1. 今天（1月7日）的24小时打车订单统计
    print("1. 生成今天24小时打车订单统计...")
    if not df_today.empty:
        generate_single_day_hourly_stats(df_today, OUTPUT_PATH, TODAY)
    else:
        print(f"警告: 无法找到日期 {TODAY} 的数据")
    
    # 2. 今天相对昨天的同期增长
    print("\n2. 生成今天相对昨天的增长对比...")
    if not df_today.empty and not df_yesterday.empty:
        generate_day_comparison_stats(df_today, df_yesterday, OUTPUT_PATH, TODAY, YESTERDAY)
    else:
        print("警告: 今天或昨天的数据不完整，无法进行对比")
    
    # 3. 最近三天汇总的24小时统计
    print("\n3. 生成最近三天汇总统计...")
    if not df_recent_3.empty:
        generate_multi_day_aggregated_stats(df_recent_3, OUTPUT_PATH, RECENT_3_DAYS)
    else:
        print("警告: 最近三天的数据不完整")
    
    # 4. 最近三天相对之前三天的增长对比
    print("\n4. 生成时期对比统计...")
    if not df_recent_3.empty and not df_previous_3.empty:
        generate_period_comparison_stats(df_recent_3, df_previous_3, OUTPUT_PATH, RECENT_3_DAYS, PREVIOUS_3_DAYS)
    else:
        print("警告: 时期对比数据不完整")
    
    # 5. 道路拥堵分析（多个时段）
    # print("\n5. 生成道路拥堵分析...")
    # if not df_today.empty:
    #     # 全天拥堵分析
    #     generate_congestion_analysis_csv(
    #         df_today, "2015-01-07 00:00:00", "2015-01-07 23:59:59", 
    #         OUTPUT_PATH, f"full_day_{TODAY}", top_n=100
    #     )
    # else:
    #     print("警告: 今天的数据不完整，无法进行拥堵分析")
    
    print(f"\n=== 分析完成！所有结果已保存到 {OUTPUT_PATH} ===")
    
if __name__ == "__main__":
    main()
