import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import logging
import os
from calc_crowdedness import api21_top_congested_roads, load_and_preprocess_csv, load_multiple_days_data

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_road_name_mapping():
    """
    从数据库获取road_id到name的映射关系
    
    返回:
        dict: {road_id: name} 映射字典，过滤掉没有name的记录
    """
    # 数据库连接配置
    DB_CONFIG = {
        'host': '106.75.3.204',
        'port': '5432',
        'database': 'harbin',
        'user': 'osmuser',
        'password': 'pass'
    }
    
    road_name_mapping = {}
    
    try:
        # 创建数据库连接
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?client_encoding=utf8"
        engine = create_engine(
            connection_string,
            connect_args={"client_encoding": "utf8"}
        )
        
        logger.info("成功连接到PostgreSQL数据库")
        
        # 查询所有有name的道路
        sql_query = """
        SELECT 
            b.gid,
            w.tags
        FROM bfmap_ways b
        INNER JOIN ways w ON b.osm_id = w.id
        WHERE w.tags IS NOT NULL
        """
        
        df_roads = pd.read_sql_query(sql_query, engine)
        logger.info(f"从数据库获取到 {len(df_roads)} 条道路记录")
        
        # 解析tags并提取name
        def extract_name_from_tags(tags_str):
            try:
                if pd.isna(tags_str) or tags_str is None:
                    return None
                import ast
                tags_dict = ast.literal_eval(str(tags_str))
                return tags_dict.get('name', None)
            except:
                return None
        
        df_roads['name'] = df_roads['tags'].apply(extract_name_from_tags)
        
        # 过滤掉没有name的记录
        df_roads_with_name = df_roads[df_roads['name'].notna()].copy()
        logger.info(f"有name的道路记录数: {len(df_roads_with_name)}")
        
        # 构建映射字典
        for _, row in df_roads_with_name.iterrows():
            road_name_mapping[row['gid']] = row['name']
        
        logger.info(f"构建了 {len(road_name_mapping)} 个road_id到name的映射")
        
    except Exception as e:
        logger.error(f"获取道路名称映射时发生错误: {str(e)}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
    
    finally:
        if 'engine' in locals():
            engine.dispose()
            logger.info("数据库连接已关闭")
    
    return road_name_mapping


def aggregate_congestion_by_name(congested_roads, road_name_mapping):
    """
    将拥堵统计按道路名称聚合
    
    参数:
        congested_roads: [(road_id, count), ...] 按road_id的拥堵统计
        road_name_mapping: {road_id: name} road_id到name的映射
    
    返回:
        list: [(name, total_count), ...] 按name聚合后的拥堵统计，按拥堵程度排序
    """
    name_congestion = {}
    
    for road_id, count in congested_roads:
        # 获取对应的道路名称
        road_name = road_name_mapping.get(road_id)
        
        # 只统计有名称的道路
        if road_name:
            if road_name in name_congestion:
                name_congestion[road_name] += count
            else:
                name_congestion[road_name] = count
    
    # 按拥堵程度排序
    sorted_name_congestion = sorted(name_congestion.items(), key=lambda x: x[1], reverse=True)
    
    logger.info(f"聚合后的道路名称数量: {len(sorted_name_congestion)}")
    if sorted_name_congestion:
        logger.info(f"最拥堵的道路: {sorted_name_congestion[0][0]} (拥堵程度: {sorted_name_congestion[0][1]})")
    
    return sorted_name_congestion


def generate_congestion_analysis_by_name_csv(df_input, time_begin_str, time_end_str, output_path, analysis_name, top_n=100):
    """
    生成按道路名称聚合的拥堵分析CSV
    
    参数:
        df_input: 输入数据DataFrame
        time_begin_str: 开始时间字符串
        time_end_str: 结束时间字符串
        output_path: 输出路径
        analysis_name: 分析名称
        top_n: 返回前N个最拥堵的道路名称
    
    返回:
        DataFrame: 分析结果
    """
    try:
        logger.info("开始拥堵分析...")
        
        # 1. 获取road_id级别的拥堵统计
        congested_roads = api21_top_congested_roads(df_input, time_begin_str, time_end_str, top_n=1000)  # 先获取更多数据
        logger.info(f"获取到 {len(congested_roads)} 条road_id级别的拥堵记录")
        
        # 2. 获取road_id到name的映射
        road_name_mapping = get_road_name_mapping()
        
        # 3. 按名称聚合拥堵统计
        name_congestion_list = aggregate_congestion_by_name(congested_roads, road_name_mapping)
        
        # 4. 取前top_n个
        top_name_congestion = name_congestion_list[:top_n]
        
        # 5. 准备CSV数据
        data = []
        for rank, (road_name, total_count) in enumerate(top_name_congestion, 1):
            # 统计该名称对应的road_id数量
            road_ids_for_name = [road_id for road_id, name in road_name_mapping.items() if name == road_name]
            
            data.append({
                'rank': rank,
                'road_name': road_name,
                'total_congestion_count': total_count,
                'road_ids_count': len(road_ids_for_name),
                'road_ids_sample': ','.join(map(str, road_ids_for_name[:5])) + ('...' if len(road_ids_for_name) > 5 else ''),
                'time_begin': time_begin_str,
                'time_end': time_end_str,
                'analysis_name': analysis_name
            })
        
        # 6. 保存CSV
        df_result = pd.DataFrame(data)
        csv_filename = f"congestion_analysis_by_name_{analysis_name}.csv"
        full_path = os.path.join(output_path, csv_filename)
        df_result.to_csv(full_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"按名称聚合的拥堵分析已保存到: {full_path}")
        
        # 7. 打印一些统计信息
        print(f"\n=== 按道路名称聚合的拥堵分析结果 ===")
        print(f"分析时间段: {time_begin_str} 到 {time_end_str}")
        print(f"原始road_id记录数: {len(congested_roads)}")
        print(f"有名称的拥堵道路数: {len(name_congestion_list)}")
        print(f"输出前 {len(top_name_congestion)} 个最拥堵道路")
        
        if top_name_congestion:
            print(f"\n前10个最拥堵道路:")
            for i, (name, count) in enumerate(top_name_congestion[:10], 1):
                print(f"  {i}. {name}: {count}")
        
        return df_result
    
    except Exception as e:
        logger.error(f"生成按名称聚合的拥堵分析CSV时出错: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return pd.DataFrame()


def main():
    """
    主函数 - 生成按道路名称聚合的拥堵分析
    """
    BASE_PATH = "/home/users/lichuan/shared"
    OUTPUT_PATH = "./screen1"
    
    # 创建输出目录
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # 定义分析日期和时间
    TODAY = "150107"
    TIME_BEGIN = "2015-01-07 00:00:00"
    TIME_END = "2015-01-07 23:59:59"
    
    print("=== 开始按道路名称聚合的拥堵分析 ===\n")
    
    # 加载今天的数据
    print("加载数据...")
    df_today = load_multiple_days_data(BASE_PATH, [TODAY], "gcj_trips")
    
    if df_today.empty:
        print(f"错误: 无法加载日期 {TODAY} 的数据")
        return
    
    print(f"成功加载 {len(df_today)} 条记录")
    
    # 生成按名称聚合的拥堵分析
    generate_congestion_analysis_by_name_csv(
        df_today, 
        TIME_BEGIN, 
        TIME_END, 
        OUTPUT_PATH, 
        f"full_day_{TODAY}_by_name", 
        top_n=100
    )
    
    print(f"\n=== 分析完成！结果已保存到 {OUTPUT_PATH} ===")


if __name__ == "__main__":
    main()
