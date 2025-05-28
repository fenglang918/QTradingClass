import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

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
    data['year_month'] = data['date'].dt.to_period('M')
    monthly = data.groupby(['first_industry_name', 'year_month'])['index_return'].sum().rename('monthly_return').to_frame()
    monthly['target_monthly_return'] = monthly.groupby(level=0)['monthly_return'].shift(-1)
    monthly = monthly.dropna(subset=['target_monthly_return'])
    
    feature_cols = [col for col in data.columns if col not in ['date', 'first_industry_name', 'index', 'index_return', 'year_month']]
    monthly_features = data.groupby(['first_industry_name', 'year_month'])[feature_cols].mean()
    
    monthly_data = monthly_features.join(monthly['target_monthly_return']).reset_index()
    monthly_data['target'] = (monthly_data['target_monthly_return'] > 0).astype(int)
    
    return monthly_data, feature_cols

# 训练模型
def train_model(monthly_data, feature_cols):
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
    print("Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    
    return model

# 计算行业权重
def calculate_weights(predictions, momentum, volatility, rsi):
    # 基础权重：预测为上涨的行业
    base_weights = predictions.copy()
    
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

# 定义Attention LSTM模型
class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_industries, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Attention层
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_industries)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # 计算attention权重
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # 应用attention
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 输出预测
        out = self.fc(context)
        return out, attention_weights

# 自定义数据集
class IndustryDataset(Dataset):
    def __init__(self, X, y, seq_length=20):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.X) - self.seq_length
        
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.seq_length]
        y = self.y[idx + self.seq_length - 1]
        return torch.FloatTensor(X_seq), torch.FloatTensor([y])

# 准备LSTM训练数据
def prepare_lstm_data(data, feature_cols, seq_length=20):
    # 按行业分组处理数据
    industry_data = {}
    for industry in data['first_industry_name'].unique():
        industry_df = data[data['first_industry_name'] == industry].copy()
        industry_df = industry_df.sort_values('date')
        
        # 准备特征
        X = industry_df[feature_cols].values
        y = (industry_df['index_return'].shift(-1) > 0).astype(float).values[:-1]  # 预测下一天涨跌
        
        # 标准化特征
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        industry_data[industry] = {
            'X': X[:-1],  # 去掉最后一天，因为没有对应的y
            'y': y,
            'scaler': scaler
        }
    
    return industry_data

# 训练LSTM模型
def train_lstm_model(industry_data, feature_cols, device='cuda' if torch.cuda.is_available() else 'cpu'):
    models = {}
    num_industries = len(industry_data)
    input_dim = len(feature_cols)
    hidden_dim = 64
    num_layers = 2
    
    for industry, data in industry_data.items():
        # 创建数据集和数据加载器
        dataset = IndustryDataset(data['X'], data['y'])
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 初始化模型
        model = AttentionLSTM(input_dim, hidden_dim, num_layers, 1).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练模型
        model.train()
        for epoch in range(50):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs, _ = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Industry: {industry}, Epoch [{epoch+1}/50], Loss: {total_loss/len(dataloader):.4f}')
        
        models[industry] = {
            'model': model,
            'scaler': data['scaler']
        }
    
    return models

# 修改回测函数以整合LSTM预测
def backtest_strategy(data, rf_model, lstm_models, feature_cols, initial_cash=1_000_000, transaction_cost=0.002):
    # 初始化
    cash = initial_cash
    positions = pd.Series(0, index=data['first_industry_name'].unique())
    net_values = []
    daily_returns = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 按日期遍历
    for date in sorted(data['date'].unique()):
        current_data = data[data['date'] == date].copy()
        
        if current_data.empty:
            continue
        
        # 获取月度模型预测
        X_day = current_data[feature_cols]
        rf_predictions = pd.Series(rf_model.predict(X_day), index=current_data['first_industry_name'])
        
        # 获取LSTM预测
        lstm_predictions = {}
        for industry in current_data['first_industry_name'].unique():
            industry_data = data[data['first_industry_name'] == industry].copy()
            industry_data = industry_data[industry_data['date'] <= date].sort_values('date')
            
            if len(industry_data) >= 20:  # 确保有足够的历史数据
                model_info = lstm_models[industry]
                X = model_info['scaler'].transform(industry_data[feature_cols].values[-20:])
                X = torch.FloatTensor(X).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    model_info['model'].eval()
                    pred, _ = model_info['model'](X)
                    lstm_predictions[industry] = torch.sigmoid(pred).item()
            else:
                lstm_predictions[industry] = 0.5  # 默认中性预测
        
        lstm_predictions = pd.Series(lstm_predictions)
        
        # 综合预测（结合RF和LSTM的预测）
        combined_predictions = (rf_predictions + (lstm_predictions > 0.5).astype(int)) / 2
        
        # 获取技术指标
        momentum = current_data.set_index('first_industry_name')['momentum_5d']
        volatility = current_data.set_index('first_industry_name')['volatility_20d']
        rsi = current_data.set_index('first_industry_name')['rsi_14d']
        
        # 计算目标权重
        target_weights = calculate_weights(combined_predictions, momentum, volatility, rsi)
        
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
    
    # 计算策略评估指标
    total_return = (net_values_df['net_value'].iloc[-1] / net_values_df['net_value'].iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(net_values_df)) - 1
    daily_returns_series = pd.Series(daily_returns)
    sharpe_ratio = np.sqrt(252) * daily_returns_series.mean() / daily_returns_series.std()
    max_drawdown = (net_values_df['net_value'] / net_values_df['net_value'].cummax() - 1).min()
    
    print("\nStrategy Performance:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annual Return: {annual_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    
    # 绘制净值曲线
    plt.figure(figsize=(12, 6))
    plt.plot(net_values_df.index, net_values_df['net_value'], label='Enhanced Strategy')
    plt.title('Enhanced Daily Sector Rotation Strategy Performance')
    plt.xlabel('Date')
    plt.ylabel('Net Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# 修改主程序
def main():
    global data  # Add global declaration
    # 计算技术指标
    data = calculate_technical_indicators(data)
    
    # 准备月度特征并训练RF模型
    monthly_data, feature_cols = prepare_monthly_features(data)
    rf_model = train_model(monthly_data, feature_cols)
    
    # 准备LSTM数据并训练模型
    print("\nTraining LSTM models...")
    industry_data = prepare_lstm_data(data, feature_cols)
    lstm_models = train_lstm_model(industry_data, feature_cols)
    
    # 筛选回测时间段
    backtest_data = data[(data['date'] >= '2020-01-01') & (data['date'] <= '2024-12-31')]
    
    # 执行回测
    print("\nRunning backtest...")
    net_values, daily_returns = backtest_strategy(backtest_data, rf_model, lstm_models, feature_cols)
    
    # 评估策略
    evaluate_strategy(net_values, daily_returns)

if __name__ == "__main__":
    main()
