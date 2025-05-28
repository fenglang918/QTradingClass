import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
index_df = pd.read_csv('industry_index_with_close2.csv', encoding='GBK', parse_dates=['date'])
# 策略：每个月月初利用上个月的所有行业日度数据作为特征变量，预测本月的30个行业是涨(1)还是跌(0)。
# 模型采用非线性分类模型，包括RF,HISTGB,SVM,K邻近和神经网络，都是直接用的sklearning的框架，调了准确率最好的RF作为实际策略
# 具体持仓策略是，每个月月初用上的月数据得到这个月行业是否涨跌，对于所有预测为涨的行业等权重以首日开盘价买入，在行业内等权配置（因为要顺着第二题和第三题来走），持有至月底收盘价卖出
# 回测数据用的是基于给的parquet数据自行计算的20200101-20241231行业平均开盘价open和收盘价close。初始仓位1000000.交易费用千2.
# 生成分类目标：下一日涨跌（涨>=0为1，否则为0）
data = index_df.copy()

# 计算月度总涨跌幅: next month total return per industry
data['year_month'] = data['date'].dt.to_period('M')
monthly = data.groupby(['first_industry_name', 'year_month'])['index_return'].sum().rename('monthly_return').to_frame()
monthly['target_monthly_return'] = monthly.groupby(level=0)['monthly_return'].shift(-1)
monthly = monthly.dropna(subset=['target_monthly_return'])

# 排除非数值列（如 year_month）
feature_cols = [col for col in data.columns if col not in ['date','first_industry_name','index','index_return','year_month']]
monthly_features = data.groupby(['first_industry_name', 'year_month'])[feature_cols].mean()

# 合并特征与标签
monthly_data = monthly_features.join(monthly['target_monthly_return']).reset_index()

# 创建二分类标签：上涨为1，否则为0
monthly_data['target'] = (monthly_data['target_monthly_return'] > 0).astype(int)

# 划分训练/测试集（2019-12 以前训练）
train_mask = monthly_data['year_month'] <= pd.Period('2019-12', freq='M')
X = monthly_data[feature_cols]
y = monthly_data['target']
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

# 删除缺失行并同步标签
def dropna_xy(X, y):
    idx = X.dropna().index
    return X.loc[idx], y.loc[idx]
X_train, y_train = dropna_xy(X_train, y_train)
X_test, y_test = dropna_xy(X_test, y_test)

# 导入分类模型与评估
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import  LogisticRegression, Lasso, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import ComplementNB, BernoulliNB
clf_models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'HistGB': HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42),
    'SVM': SVC(kernel='rbf', C=1.0, probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Neural Network':MLPClassifier()
}

# 训练与评估
for name, model in clf_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")

# --------------------------- 随机森林 策略回测---------------------------
import matplotlib.pyplot as plt
import numpy as np

# 确保 index_df 中包含 'open' 和 'close' 列
if 'open' not in index_df.columns or 'close' not in index_df.columns:
    raise ValueError("index_df 缺少 'open' 或 'close' 列，无法进行回测。")

# 准备数据
data = index_df.copy()
data['year_month'] = data['date'].dt.to_period('M')

# 计算月度特征
feature_cols = [col for col in data.columns if
                col not in ['date', 'first_industry_name', 'index', 'index_return', 'year_month']]
monthly_features = data.groupby(['first_industry_name', 'year_month'])[feature_cols].mean().reset_index()

# 生成目标变量
monthly_returns = data.groupby(['first_industry_name', 'year_month'])['index_return'].sum().rename(
    'monthly_return').reset_index()
monthly_returns['target_monthly_return'] = monthly_returns.groupby('first_industry_name')['monthly_return'].shift(-1)
monthly_returns = monthly_returns.dropna(subset=['target_monthly_return'])
monthly_returns['target'] = (monthly_returns['target_monthly_return'] > 0).astype(int)

# 合并特征与标签
monthly_data = pd.merge(monthly_features, monthly_returns[['first_industry_name', 'year_month', 'target']],
                        on=['first_industry_name', 'year_month'], how='inner')

# 筛选回测时间段
monthly_data['year_month'] = monthly_data['year_month'].astype(str)
backtest_data = monthly_data[(monthly_data['year_month'] >= '2020-01') & (monthly_data['year_month'] <= '2024-12')]

# 初始化净值
initial_cash = 1_000_000
cash = initial_cash
net_values = []

# 获取每月的第一个和最后一个交易日
data['year_month'] = data['date'].dt.to_period('M')
first_days = data.groupby('year_month')['date'].min()
last_days = data.groupby('year_month')['date'].max()

# 进行回测
for ym in sorted(backtest_data['year_month'].unique()):
    current_month = pd.Period(ym)
    first_day = first_days.get(current_month)
    last_day = last_days.get(current_month)

    if pd.isna(first_day) or pd.isna(last_day):
        continue

    # 获取当月特征数据
    month_features = backtest_data[backtest_data['year_month'] == ym]
    X_month = month_features[feature_cols]

    # 预测上涨行业
    predicted = clf_models['RandomForest'].predict(X_month)
    up_indices = month_features[predicted == 1]['first_industry_name'].values
    down_ratio = (predicted == 0).sum() / len(predicted)

    # 获取开盘价和收盘价
    open_prices = data[(data['date'] == first_day) & (data['first_industry_name'].isin(up_indices))][
        ['first_industry_name', 'open']]
    close_prices = data[(data['date'] == last_day) & (data['first_industry_name'].isin(up_indices))][
        ['first_industry_name', 'close']]

    if open_prices.empty or close_prices.empty:
        net_values.append({'date': last_day, 'net_value': cash})
        continue

    # 合并开盘价和收盘价
    prices = pd.merge(open_prices, close_prices, on='first_industry_name', how='inner')
    prices = prices.dropna()

    if prices.empty:
        net_values.append({'date': last_day, 'net_value': cash})
        continue

    # 计算每个行业的资金分配
    num_industries = len(prices)
    allocation = cash / num_industries
    returns = (prices['close'] - prices['open']) / prices['open']
    cash = (allocation * (1 + returns - 0.002)).sum()
    net_values.append({'date': last_day, 'net_value': cash})

# 创建净值曲线 DataFrame
net_value_df = pd.DataFrame(net_values)
net_value_df.set_index('date', inplace=True)

# 绘制净值曲线
plt.figure(figsize=(12, 6))
plt.plot(net_value_df.index, net_value_df['net_value'], label='RF')
plt.title('Random Forest net value')
plt.xlabel('date')
plt.ylabel('net value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
