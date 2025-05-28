import pandas as pd

# 读取数据
df = pd.read_parquet('data.parquet')

# 确保日期类型，并排序
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['first_industry_name', 'code', 'date'])

# 标记每只股票是否在前一交易日存在
df['prev_date'] = df.groupby('code')['date'].shift(1)
df['exist'] = df['prev_date'].notna().astype(int)

# 标记每只股票是否在下一交易日存在
df['next_date'] = df.groupby('code')['date'].shift(-1)
df['exist2'] = df['next_date'].notna().astype(int)

# 筛选时间范围
mask = (df['date'] >= '2006-01-01') & (df['date'] <= '2024-12-31')
filtered = df[mask]

results = []

# 提取附加指标列（从 ret_o2c 开始的列）
extra_cols = [col for col in df.columns if col.startswith('ret_') or col.endswith('_ttm') or col.endswith('_lf') or col in ['book_to_market_ratio_1f', 'dividend_yield_ttm']]

# 按行业和日期分组计算指标和指数
for industry, industry_group in filtered.groupby('first_industry_name'):
    date_list = sorted(industry_group['date'].unique())
    prev_index = 100.0
    prev_close3 = None  # 用于索引计算的上一日 close3

    for date in date_list:
        # today 表示当前行业在该日期的全部个股日度数据
        today = industry_group[industry_group['date'] == date]
        close_mean = today['close'].mean()

        # close2: 只对 exist==1 的记录计算均值
        close2 = today.loc[today['exist'] == 1, 'close'].mean()

        # close3: 只对 exist2==1 的记录计算均值
        close3 = today.loc[today['exist2'] == 1, 'close'].mean()

        # 计算指数: 初始100，后续 = prev_index * close2 / prev_close3
        if prev_close3 is not None and pd.notna(close2) and prev_close3 != 0:
            idx_val = prev_index * close2 / prev_close3
        else:
            idx_val = 100.0

        # 聚合其他指标
        open_mean = today['open'].mean()
        high_mean = today['high'].mean()
        low_mean = today['low'].mean()
        volume_sum = today['volume'].sum()

        # 汇总附加财务指标
        extra_metrics = {col: today[col].mean() for col in extra_cols if col in today.columns}

        row = {
            'first_industry_name': industry,
            'date': date,
            'open': open_mean,
            'high': high_mean,
            'low': low_mean,
            'close': close_mean,
            'close2': close2,
            'close3': close3,
            'exist_rate': today['exist'].mean(),  # 当日前一日存在率
            'exist2_rate': today['exist2'].mean(),  # 当日下一日存在率
            'volume': volume_sum,
            'index': idx_val,
        }

        row.update(extra_metrics)  # 添加财务指标
        results.append(row)

        # 更新缓存
        prev_index = idx_val
        prev_close3 = close3

# 汇总结果并设置索引名称
index_df = pd.DataFrame(results)
index_df = index_df.sort_values(['first_industry_name', 'date'])

# 计算每个行业指数的涨跌幅
index_df['index_return'] = index_df.groupby('first_industry_name')['index'].pct_change().fillna(0)

# 导出CSV
index_df.to_csv('industry_index_with_close2.csv', index=False, encoding='GBK')
print(index_df.head())