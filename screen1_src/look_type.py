import pandas as pd


def get_csv_field_types_postgresql(csv_file_path):
    """获取CSV文件各字段的PostgreSQL数据类型"""
    df = pd.read_csv(csv_file_path)
    
    print(f"文件: {csv_file_path}")
    print("PostgreSQL字段类型:")
    print("-" * 40)
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        # 转换为PostgreSQL类型
        if dtype == 'int64':
            pg_type = 'INTEGER'
        elif dtype == 'float64':
            pg_type = 'NUMERIC(10,2)'
        elif dtype == 'object':
            max_len = df[col].astype(str).str.len().max()
            if max_len <= 255:
                pg_type = f'VARCHAR({max_len + 10})'
            else:
                pg_type = 'TEXT'
        elif 'datetime' in dtype:
            pg_type = 'TIMESTAMP'
        elif dtype == 'bool':
            pg_type = 'BOOLEAN'
        else:
            pg_type = 'TEXT'
            
        print(f"{col}: {dtype} -> {pg_type}")
    
    print()

# 使用示例
csv_files = [
    "/home/users/lichuan/projects/sjzt/screen1/aggregated_hourly_trips_150105_to_150107.csv",
    "/home/users/lichuan/projects/sjzt/screen1/congestion_analysis_full_day_150107.csv",
    "/home/users/lichuan/projects/sjzt/screen1/growth_comparison_150107_vs_150106.csv",
    "/home/users/lichuan/projects/sjzt/screen1/hourly_trips_150107.csv",
    "/home/users/lichuan/projects/sjzt/screen1/period_comparison_150105_to_150107_vs_150103_to_150105.csv",
    "/home/users/lichuan/projects/sjzt/screen1/cluster_centers_flow.csv",
]

for file in csv_files:
    try:
        get_csv_field_types_postgresql(file)
    except FileNotFoundError:
        print(f"文件未找到: {file}\n")
