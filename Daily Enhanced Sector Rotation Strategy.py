import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 确保输入维度正确 [batch_size, seq_length, features]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # 添加batch维度
        elif x.dim() == 4:
            batch_size, samples, seq_len, features = x.size()
            x = x.view(-1, seq_len, features)
        
        # LSTM层
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_length, hidden_size*2]
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)  # [batch_size, seq_length, 1]
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_size*2]
        
        # 全连接层
        out = self.fc(context)  # [batch_size, 1]
        
        return out.squeeze(-1)  # 返回 [batch_size]

# 定义数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_length=20):
        # 确保输入数据为float32类型
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.X) - self.seq_length
    
    def __getitem__(self, idx):
        # 返回形状为 [seq_length, features] 的序列
        X_seq = self.X[idx:idx + self.seq_length]
        y = self.y[idx + self.seq_length - 1]
        return torch.from_numpy(X_seq).float(), torch.tensor(y, dtype=torch.float32)

def prepare_sequences(data, feature_cols, seq_length=20):
    """
    准备时序序列数据，确保数据类型为float32
    """
    sequences = []
    targets = []
    
    # 确保只使用数值型特征列
    numeric_cols = data[feature_cols].select_dtypes(include=[np.number]).columns
    if len(numeric_cols) != len(feature_cols):
        print(f"\n警告：部分特征列不是数值类型，已自动过滤。")
        print(f"原始特征数：{len(feature_cols)}")
        print(f"数值型特征数：{len(numeric_cols)}")
        print("非数值型特征：", set(feature_cols) - set(numeric_cols))
    
    for industry in data['first_industry_name'].unique():
        industry_data = data[data['first_industry_name'] == industry].sort_values('date')
        # 只使用数值型特征列
        X = industry_data[numeric_cols].astype(np.float32).values
        y = industry_data['y'].astype(np.float32).values
        
        for i in range(len(X) - seq_length + 1):
            sequences.append(X[i:i + seq_length])
            targets.append(y[i + seq_length - 1])
    
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

def train_lstm_model(X, y, train_mask, val_mask, params=None):
    """
    训练LSTM模型，使用时序划分
    """
    if params is None:
        params = dict(
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            batch_size=32,
            learning_rate=0.001,
            num_epochs=50,
            seq_length=20,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    print(f"\n使用设备: {params['device']}")
    
    # 准备序列数据
    X_seq, y_seq = prepare_sequences(X, X.columns, params['seq_length'])
    
    # 使用时序mask划分训练集和验证集
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    
    # 确保序列完整性
    train_indices = train_indices[train_indices + params['seq_length'] - 1 < len(X_seq)]
    val_indices = val_indices[val_indices + params['seq_length'] - 1 < len(X_seq)]
    
    X_train = X_seq[train_indices]
    y_train = y_seq[train_indices]
    X_val = X_seq[val_indices]
    y_val = y_seq[val_indices]
    
    print(f"\n训练集形状：X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"验证集形状：X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    # 创建数据加载器
    train_dataset = TimeSeriesDataset(X_train, y_train, params['seq_length'])
    val_dataset = TimeSeriesDataset(X_val, y_val, params['seq_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    
    # 初始化模型
    model = LSTMModel(
        input_size=X_seq.shape[2],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(params['device'])
    
    print(f"\n模型输入维度：{X_seq.shape[2]}")
    print(f"模型隐藏层维度：{params['hidden_size']}")
    print(f"模型层数：{params['num_layers']}")
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # 训练模型
    best_val_auc = 0
    best_model = None
    early_stopping_counter = 0
    
    print("\n开始训练LSTM模型...")
    for epoch in range(params['num_epochs']):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(params['device'])
            batch_y = batch_y.to(params['device'])
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())
        
        # 验证
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(params['device'])
                outputs = model(batch_X)
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(batch_y.numpy())
        
        # 计算指标
        train_auc = roc_auc_score(train_labels, train_preds)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        print(f"Epoch {epoch+1}/{params['num_epochs']}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Train AUC: {train_auc:.4f}")
        print(f"Val AUC: {val_auc:.4f}")
        
        # 学习率调整
        scheduler.step(val_auc)
        
        # 早停
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model.state_dict().copy()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= 10:
                print("Early stopping triggered")
                break
    
    # 加载最佳模型
    model.load_state_dict(best_model)
    
    # 保存模型
    torch.save(model.state_dict(), 'lstm_model.pth')
    
    return model, params

def predict_with_lstm(model, data, feature_cols, params):
    """
    使用LSTM模型进行预测，返回每个时间点的预测值
    """
    model.eval()
    predictions = []
    dates = []
    industries = []
    
    for industry in data['first_industry_name'].unique():
        industry_data = data[data['first_industry_name'] == industry].sort_values('date')
        X = industry_data[feature_cols].values
        
        # 准备序列数据
        sequences = []
        for i in range(len(X) - params['seq_length'] + 1):
            sequences.append(X[i:i + params['seq_length']])
        
        if len(sequences) == 0:
            continue
        
        sequences = np.array(sequences, dtype=np.float32)
        
        # 预测
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequences).to(params['device'])
            preds = model(seq_tensor).cpu().numpy()
            
            # 记录预测值和对应的日期、行业
            for i, pred in enumerate(preds):
                date = industry_data.iloc[i + params['seq_length'] - 1]['date']
                predictions.append(pred)
                dates.append(date)
                industries.append(industry)
    
    # 创建预测结果DataFrame
    pred_df = pd.DataFrame({
        'date': dates,
        'first_industry_name': industries,
        'lstm_pred': predictions
    })
    
    # 合并回原始数据
    data = data.merge(pred_df, on=['date', 'first_industry_name'], how='left')
    
    return data

# 读取数据
index_df = pd.read_csv('industry_index_with_close2.csv', encoding='GBK', parse_dates=['date'])

# 数据预处理
data = index_df.copy()

# 计算技术指标
def calculate_technical_indicators(df):
    # 计算动量因子 (5日收益率)
    df['momentum_5d'] = df.groupby('first_industry_name')['index_return'].transform(
        lambda x: x.rolling(5).sum().shift(1))
    
    # 计算波动率因子 (20日波动率)
    df['volatility_20d'] = df.groupby('first_industry_name')['index_return'].transform(
        lambda x: x.rolling(20).std().shift(1))
    
    # 计算RSI (14日)
    delta = df.groupby('first_industry_name')['index_return'].transform(lambda x: x.diff())
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14d'] = 100 - (100 / (1 + rs))
    
    return df

# 计算月度特征（用于模型训练）
def prepare_monthly_features(data):
    """
    准备月度特征并训练RF模型
    """
    data['year_month'] = data['date'].dt.to_period('M')
    monthly = data.groupby(['first_industry_name', 'year_month'])['index_return'].sum().rename('monthly_return').to_frame()
    monthly['target_monthly_return'] = monthly.groupby(level=0)['monthly_return'].shift(-1)
    monthly = monthly.dropna(subset=['target_monthly_return'])
    
    feature_cols = [col for col in data.columns if col not in ['date', 'first_industry_name', 'index', 'index_return', 'year_month']]
    monthly_features = data.groupby(['first_industry_name', 'year_month'])[feature_cols].mean()
    
    monthly_data = monthly_features.join(monthly['target_monthly_return']).reset_index()
    monthly_data['target'] = (monthly_data['target_monthly_return'] > 0).astype(int)
    
    # 训练RF模型并生成预测
    train_mask = monthly_data['year_month'] <= pd.Period('2019-12', freq='M')
    X = monthly_data[feature_cols]
    y = monthly_data['target']
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    
    # 删除缺失值
    X_train = X_train.dropna()
    y_train = y_train[X_train.index]
    X_test = X_test.dropna()
    y_test = y_test[X_test.index]
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    print("Monthly RF Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    
    # 生成月度预测信号
    monthly_data['rf_pred'] = model.predict(X)
    
    return monthly_data, feature_cols, model

# 计算行业权重
def calculate_weights(rf_pred, lstm_pred, momentum, volatility, rsi, rf_weight=0.4, lstm_weight=0.6):
    """
    计算综合权重，考虑RF和LSTM的预测信号
    """
    # 基础权重：综合RF和LSTM的预测
    base_weights = rf_pred * rf_weight + lstm_pred * lstm_weight
    
    # 根据动量调整权重
    momentum_weights = momentum.rank(pct=True)
    
    # 根据波动率调整权重（低波动率获得更高权重）
    volatility_weights = (1 / volatility).rank(pct=True)
    
    # 根据RSI调整权重（RSI在40-60之间获得更高权重）
    rsi_weights = 1 - abs(rsi - 50) / 50
    
    # 综合权重
    final_weights = base_weights * momentum_weights * volatility_weights * rsi_weights
    
    # 归一化权重
    if final_weights.sum() > 0:
        final_weights = final_weights / final_weights.sum()
    
    return final_weights

# 修改回测函数以整合LSTM预测
def backtest_strategy(data, rf_model, feature_cols, initial_cash=1_000_000, transaction_cost=0.002):
    """
    修改后的回测函数，整合RF和LSTM的预测信号
    """
    # 初始化
    cash = initial_cash
    positions = pd.Series(0, index=data['first_industry_name'].unique())
    net_values = []
    daily_returns = []
    
    # 按日期遍历
    for date in sorted(data['date'].unique()):
        current_data = data[data['date'] == date].copy()
        
        if current_data.empty:
            continue
        
        # 获取RF模型预测
        X_day = current_data[feature_cols]
        rf_predictions = pd.Series(rf_model.predict(X_day), index=current_data['first_industry_name'])
        
        # 获取LSTM预测
        lstm_predictions = current_data.set_index('first_industry_name')['lstm_pred']
        
        # 获取技术指标
        momentum = current_data.set_index('first_industry_name')['momentum_5d']
        volatility = current_data.set_index('first_industry_name')['volatility_20d']
        rsi = current_data.set_index('first_industry_name')['rsi_14d']
        
        # 计算目标权重
        target_weights = calculate_weights(
            rf_predictions, 
            lstm_predictions,
            momentum, 
            volatility, 
            rsi
        )
        
        # 计算当前持仓市值
        current_prices = current_data.set_index('first_industry_name')['close']
        current_value = (positions * current_prices).sum()
        total_value = cash + current_value
        
        # 计算目标持仓
        target_positions = (target_weights * total_value / current_prices).fillna(0)
        
        # 计算交易成本
        trades = target_positions - positions
        trade_cost = abs(trades * current_prices * transaction_cost).sum()
        
        # 更新持仓和现金
        positions = target_positions
        cash = total_value - (positions * current_prices).sum() - trade_cost
        
        # 记录净值
        net_value = cash + (positions * current_prices).sum()
        net_values.append({'date': date, 'net_value': net_value})
        
        # 计算日收益率
        if len(net_values) > 1:
            daily_return = (net_value / net_values[-2]['net_value']) - 1
            daily_returns.append(daily_return)
    
    return pd.DataFrame(net_values), daily_returns

def evaluate_strategy(net_values, daily_returns):
    net_values_df = pd.DataFrame(net_values)
    net_values_df.set_index('date', inplace=True)
    
    # 创建保存图片的目录
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 使用当前时间创建子目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(results_dir, f'backtest_{timestamp}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 计算策略评估指标
    total_return = (net_values_df['net_value'].iloc[-1] / net_values_df['net_value'].iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(net_values_df)) - 1
    daily_returns_series = pd.Series(daily_returns)
    sharpe_ratio = np.sqrt(252) * daily_returns_series.mean() / daily_returns_series.std()
    max_drawdown = (net_values_df['net_value'] / net_values_df['net_value'].cummax() - 1).min()
    
    # 计算月度收益率
    monthly_returns = net_values_df['net_value'].resample('ME').last().pct_change()
    monthly_returns = monthly_returns[monthly_returns.notna()]
    
    # 计算年化波动率
    annual_volatility = daily_returns_series.std() * np.sqrt(252)
    
    # 计算最大回撤持续时间
    cummax = net_values_df['net_value'].cummax()
    drawdown = (net_values_df['net_value'] / cummax - 1)
    max_drawdown_duration = (drawdown == 0).astype(int).groupby((drawdown != 0).cumsum()).cumsum().max()
    
    # 保存策略评估指标到文本文件
    with open(os.path.join(save_dir, 'strategy_performance.txt'), 'w') as f:
        f.write("Strategy Performance:\n")
        f.write(f"Total Return: {total_return:.2%}\n")
        f.write(f"Annual Return: {annual_return:.2%}\n")
        f.write(f"Annual Volatility: {annual_volatility:.2%}\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")
        f.write(f"Maximum Drawdown: {max_drawdown:.2%}\n")
        f.write(f"Max Drawdown Duration: {max_drawdown_duration} days\n")
        f.write(f"Monthly Win Rate: {(monthly_returns > 0).mean():.2%}\n")
    
    print("\nStrategy Performance:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annual Return: {annual_return:.2%}")
    print(f"Annual Volatility: {annual_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Max Drawdown Duration: {max_drawdown_duration} days")
    print(f"Monthly Win Rate: {(monthly_returns > 0).mean():.2%}")
    
    # 创建图表
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 净值曲线
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(net_values_df.index, net_values_df['net_value'], label='Strategy NAV', color='#1f77b4')
    ax1.plot(net_values_df.index, net_values_df['net_value'].cummax(), '--', label='Historical High', color='#ff7f0e', alpha=0.5)
    ax1.set_title('Strategy Net Asset Value')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Net Value')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 回撤曲线
    ax2 = plt.subplot(2, 2, 2)
    drawdown.plot(ax=ax2, color='#d62728', alpha=0.7)
    ax2.set_title('Drawdown Curve')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 月度收益率热力图
    ax3 = plt.subplot(2, 2, 3)
    monthly_returns_matrix = monthly_returns.to_frame()
    monthly_returns_matrix.index = pd.MultiIndex.from_arrays([
        monthly_returns_matrix.index.year,
        monthly_returns_matrix.index.month
    ])
    monthly_returns_matrix = monthly_returns_matrix.unstack()
    
    im = ax3.imshow(monthly_returns_matrix, cmap='RdYlGn', aspect='auto')
    ax3.set_title('Monthly Returns Heatmap')
    plt.colorbar(im, ax=ax3, format='%.1%')
    
    years = sorted(monthly_returns_matrix.index.unique())
    months = range(1, 13)
    ax3.set_xticks(range(len(months)))
    ax3.set_yticks(range(len(years)))
    ax3.set_xticklabels(months)
    ax3.set_yticklabels(years)
    
    for i in range(len(years)):
        for j in range(len(months)):
            if (years[i], months[j]) in monthly_returns_matrix.index:
                value = monthly_returns_matrix.loc[(years[i], months[j])].iloc[0]
                color = 'white' if abs(value) > 0.1 else 'black'
                ax3.text(j, i, f'{value:.1%}', ha='center', va='center', color=color)
    
    # 4. 收益率分布
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(daily_returns_series, bins=50, color='#2ca02c', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax4.set_title('Daily Returns Distribution')
    ax4.set_xlabel('Return')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存组合图表
    plt.savefig(os.path.join(save_dir, 'strategy_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存月度统计
    monthly_stats = pd.DataFrame({
        'Return': monthly_returns,
        'Cumulative Return': (1 + monthly_returns).cumprod() - 1
    })
    monthly_stats.to_csv(os.path.join(save_dir, 'monthly_statistics.csv'))
    
    # 保存年度统计
    yearly_returns = net_values_df['net_value'].resample('Y').last().pct_change()
    yearly_stats = pd.DataFrame({
        'Year': [year.year for year in yearly_returns.index],
        'Return': yearly_returns.values
    })
    yearly_stats.to_csv(os.path.join(save_dir, 'yearly_statistics.csv'), index=False)
    
    print(f"\nResults saved to directory: {save_dir}")
    print("\nMonthly Returns Statistics:")
    print(monthly_stats.describe())
    
    print("\nAnnual Returns:")
    for year, ret in yearly_returns.items():
        if not pd.isna(ret):
            print(f"{year.year}: {ret:.2%}")
            
    return save_dir  # 返回save_dir供main函数使用

def add_financial_features(df):
    """
    添加财务指标特征，只处理实际存在的财务指标列
    """
    # 获取所有可能的财务指标列
    financial_cols = [col for col in df.columns if col.endswith('_ttm') or col.endswith('_lf') or 
                     col in ['book_to_market_ratio_1f', 'dividend_yield_ttm']]
    
    print("\n可用的财务指标列：")
    print(financial_cols)
    
    # 对每个存在的财务指标进行处理
    for col in financial_cols:
        if col in df.columns:
            # 计算排名
            df[f'{col}_rank'] = df.groupby('date')[col].transform(lambda x: x.rank(pct=True))
            
            # 计算变化率，明确指定fill_method=None
            df[f'{col}_change'] = df.groupby('first_industry_name')[col].pct_change(fill_method=None)
            df[f'{col}_change'] = df[f'{col}_change'].fillna(0)  # 填充第一个值的NA
            
            df[f'{col}_change_ma'] = df.groupby('first_industry_name')[f'{col}_change'].transform(
                lambda x: x.rolling(20).mean()
            )
            
            print(f"已处理财务指标: {col}")
        else:
            print(f"警告: 财务指标 {col} 不存在于数据中")
    
    # 计算财务指标之间的相关性
    if len(financial_cols) > 1:
        print("\n计算财务指标相关性...")
        financial_ranks = [f'{col}_rank' for col in financial_cols if f'{col}_rank' in df.columns]
        if financial_ranks:
            corr_matrix = df[financial_ranks].corr()
            print("\n财务指标相关性矩阵：")
            print(corr_matrix)
    
    return df

def add_price_features(df):
    """
    添加更多价格相关特征
    """
    # 使用close2计算更准确的收益率，明确指定fill_method=None
    df['return_close2'] = df.groupby('first_industry_name')['close2'].pct_change(fill_method=None)
    df['return_close2'] = df['return_close2'].fillna(0)  # 填充第一个值的NA
    
    # 考虑股票连续性的波动率
    df['vol_close2'] = df.groupby('first_industry_name')['return_close2'].transform(
        lambda x: x.rolling(20).std()
    )
    
    # 考虑股票未来存在性的动量
    df['mom_close3'] = df.groupby('first_industry_name')['close3'].transform(
        lambda x: x / x.shift(20) - 1
    )
    
    # 计算close2和close3的价差
    df['close2_close3_spread'] = (df['close2'] - df['close3']) / df['close2']
    df['close2_close3_spread'] = df['close2_close3_spread'].fillna(0)  # 填充可能的NA值
    
    # 计算close2和close3的价差变化，明确指定fill_method=None
    df['close2_close3_spread_change'] = df.groupby('first_industry_name')['close2_close3_spread'].pct_change(fill_method=None)
    df['close2_close3_spread_change'] = df['close2_close3_spread_change'].fillna(0)  # 填充第一个值的NA
    
    # 计算close2和close3的价差移动平均
    df['close2_close3_spread_ma'] = df.groupby('first_industry_name')['close2_close3_spread'].transform(
        lambda x: x.rolling(20).mean()
    )
    
    return df

def add_quality_features(df):
    """
    添加股票质量相关特征
    """
    # 股票连续性指标
    df['exist_ma'] = df.groupby('first_industry_name')['exist_rate'].transform(
        lambda x: x.rolling(20).mean()
    )
    
    # 股票未来存在性指标
    df['exist2_ma'] = df.groupby('first_industry_name')['exist2_rate'].transform(
        lambda x: x.rolling(20).mean()
    )
    
    # 综合质量得分
    df['quality_score'] = (
        df['exist_ma'].fillna(0) * 0.5 + 
        df['exist2_ma'].fillna(0) * 0.5
    )
    
    # 计算质量得分的变化，明确指定fill_method=None
    df['quality_score_change'] = df.groupby('first_industry_name')['quality_score'].pct_change(fill_method=None)
    df['quality_score_change'] = df['quality_score_change'].fillna(0)  # 填充第一个值的NA
    
    # 计算质量得分的移动平均
    df['quality_score_ma'] = df.groupby('first_industry_name')['quality_score'].transform(
        lambda x: x.rolling(20).mean()
    )
    
    return df

def add_daily_features(df):
    """
    为日频数据添加更多技术指标特征
    """
    # 原有的特征计算
    g = df.groupby('first_industry_name')['close']
    
    # 动量因子
    for window in [5, 10, 20, 60]:
        df[f'ret_{window}'] = g.pct_change(window)
        df[f'vol_{window}'] = g.pct_change().rolling(window).std()
        df[f'mom_{window}'] = g.transform(lambda x: x / x.shift(window) - 1)
    
    # 波动率因子
    df['vol_ratio'] = df['vol_20'] / df['vol_60']  # 短期/长期波动率比
    
    # 趋势因子
    df['ma_5'] = g.transform(lambda x: x.rolling(5).mean())
    df['ma_20'] = g.transform(lambda x: x.rolling(20).mean())
    df['ma_60'] = g.transform(lambda x: x.rolling(60).mean())
    df['trend_5_20'] = df['ma_5'] / df['ma_20'] - 1
    df['trend_20_60'] = df['ma_20'] / df['ma_60'] - 1
    
    # 极值因子
    high_250 = g.transform(lambda x: x.rolling(250).max())
    low_250 = g.transform(lambda x: x.rolling(250).min())
    df['dist_max_250'] = (df['close'] / high_250) - 1
    df['dist_min_250'] = (df['close'] / low_250) - 1
    df['price_position'] = (df['close'] - low_250) / (high_250 - low_250)
    
    # RSI系列
    for window in [6, 14, 30]:
        delta = g.pct_change()
        up = delta.clip(lower=0).rolling(window).mean()
        dn = (-delta.clip(upper=0)).rolling(window).mean()
        rs = up / dn
        df[f'rsi_{window}'] = 100 - 100 / (1 + rs)
    
    # MACD
    exp1 = g.transform(lambda x: x.ewm(span=12, adjust=False).mean())
    exp2 = g.transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 成交量相关特征
    if 'volume' in df.columns:
        df['volume_ma5'] = df.groupby('first_industry_name')['volume'].transform(lambda x: x.rolling(5).mean())
        df['volume_ma20'] = df.groupby('first_industry_name')['volume'].transform(lambda x: x.rolling(20).mean())
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    # 交互特征
    df['mom_rsi'] = df['ret_5'] * df['rsi_14']
    df['vol_trend'] = df['vol_ratio'] * df['trend_5_20']
    df['rsi_trend'] = df['rsi_14'] * df['trend_20_60']
    
    # 行业相对强度
    df['industry_rank'] = df.groupby('date')['ret_20'].transform(lambda x: x.rank(pct=True))
    df['industry_vol_rank'] = df.groupby('date')['vol_20'].transform(lambda x: x.rank(pct=True))
    
    return df

def prepare_daily_ml_data(data, monthly_data):
    """
    准备日频机器学习数据，优化标签生成方式，避免数据泄露
    """
    # 添加所有特征
    print("1. 添加技术指标特征...")
    data = add_daily_features(data)
    
    print("\n2. 添加财务特征...")
    data = add_financial_features(data)
    
    print("\n3. 添加价格特征...")
    data = add_price_features(data)
    
    print("\n4. 添加质量特征...")
    data = add_quality_features(data)
    
    # 合并月度RF信号
    data['year_month'] = data['date'].dt.to_period('M')
    data = data.merge(
        monthly_data[['first_industry_name', 'year_month', 'rf_pred']],
        on=['first_industry_name', 'year_month'],
        how='left'
    )
    
    # 生成标签：使用未来5日收益率，但避免跨行业数据泄露
    data['future_ret_5d'] = data.groupby('first_industry_name')['index_return'].transform(
        lambda x: x.shift(-5).rolling(5).sum()
    )
    
    # 对每个行业单独计算历史分位数，避免使用未来信息
    data['historical_rank'] = data.groupby('first_industry_name')['future_ret_5d'].transform(
        lambda x: x.rolling(504, min_periods=120).apply(
            lambda y: pd.Series(y[:-1]).rank(pct=True).iloc[-1] if len(y) > 1 else np.nan
        )
    )
    
    # 生成标签：使用更温和的分位数阈值
    data['y'] = (data['historical_rank'] > 0.6).astype(int)
    
    # 删除中间计算列，避免数据泄露
    data = data.drop(['future_ret_5d', 'historical_rank'], axis=1)
    
    # 获取所有实际存在的特征列
    all_possible_features = [
        # 技术指标特征
        'ret_5', 'ret_10', 'ret_20', 'ret_60',
        'mom_5', 'mom_10', 'mom_20', 'mom_60',
        'vol_5', 'vol_10', 'vol_20', 'vol_60',
        'vol_ratio', 'trend_5_20', 'trend_20_60',
        'rsi_6', 'rsi_14', 'rsi_30',
        'macd', 'macd_signal', 'macd_hist',
        'price_position', 'dist_max_250', 'dist_min_250',
        'industry_rank', 'industry_vol_rank',
        'mom_rsi', 'vol_trend', 'rsi_trend',
        
        # 财务特征（动态获取）
        *[f'{col}_rank' for col in data.columns if col.endswith('_ttm') or col.endswith('_lf')],
        *[f'{col}_change' for col in data.columns if col.endswith('_ttm') or col.endswith('_lf')],
        *[f'{col}_change_ma' for col in data.columns if col.endswith('_ttm') or col.endswith('_lf')],
        
        # 价格特征
        'return_close2', 'vol_close2', 'mom_close3',
        'close2_close3_spread', 'close2_close3_spread_change', 'close2_close3_spread_ma',
        
        # 质量特征
        'exist_ma', 'exist2_ma', 'quality_score',
        'quality_score_change', 'quality_score_ma',
        
        # 月度模型信号
        'rf_pred'
    ]
    
    # 只保留实际存在的特征列
    feature_cols = [col for col in all_possible_features if col in data.columns]
    
    print("\n实际使用的特征列：")
    print(feature_cols)
    print(f"\n特征总数：{len(feature_cols)}")
    
    # 清理数据
    ml_df = data.dropna(subset=feature_cols + ['y']).copy()
    
    # 划分训练集、验证集和测试集
    train_end = pd.Timestamp('2019-12-31')
    val_end = pd.Timestamp('2020-12-31')
    
    train_mask = ml_df['date'] <= train_end
    val_mask = (ml_df['date'] > train_end) & (ml_df['date'] <= val_end)
    test_mask = ml_df['date'] > val_end
    
    print("\n数据集划分情况：")
    print(f"训练集：{ml_df[train_mask]['date'].min()} 到 {ml_df[train_mask]['date'].max()}")
    print(f"验证集：{ml_df[val_mask]['date'].min()} 到 {ml_df[val_mask]['date'].max()}")
    print(f"测试集：{ml_df[test_mask]['date'].min()} 到 {ml_df[test_mask]['date'].max()}")
    print(f"训练集样本数：{train_mask.sum()}")
    print(f"验证集样本数：{val_mask.sum()}")
    print(f"测试集样本数：{test_mask.sum()}")
    
    # 输出标签分布
    print("\n标签分布：")
    print(ml_df['y'].value_counts(normalize=True))
    
    return ml_df, feature_cols, train_mask, val_mask, test_mask

def main():
    global data
    
    # 读取数据
    data = pd.read_csv('industry_index_with_close2.csv', encoding='GBK', parse_dates=['date'])
    
    # 计算技术指标
    data = add_daily_features(data)
    
    # 准备月度特征
    monthly_data, monthly_feature_cols, rf_model = prepare_monthly_features(data)
    
    # 准备机器学习数据
    ml_df, feature_cols, train_mask, val_mask, test_mask = prepare_daily_ml_data(data, monthly_data)
    
    # 训练LSTM模型
    lstm_params = dict(
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=50,
        seq_length=20,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model, params = train_lstm_model(ml_df, ml_df['y'], train_mask, val_mask, lstm_params)
    
    # 生成预测并合并回数据
    ml_df = predict_with_lstm(model, ml_df, feature_cols, params)
    
    # 运行回测
    net_values, daily_returns = backtest_strategy(ml_df, rf_model, feature_cols)
    
    # 评估策略
    evaluate_strategy(net_values, daily_returns)
    
    # 保存模型架构图
    try:
        from torchviz import make_dot
        x = torch.randn(1, params['seq_length'], len(feature_cols)).to(params['device'])
        y = model(x)
        dot = make_dot(y, params=dict(model.named_parameters()))
        dot.render("lstm_model_architecture", format="png")
        print("\n模型架构图已保存为 lstm_model_architecture.png")
    except:
        print("\n无法生成模型架构图，请确保已安装torchviz包")

if __name__ == "__main__":
    main()
