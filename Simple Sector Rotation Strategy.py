import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SimpleSectorRotation:
    def __init__(self, data_path, 
                 momentum_window=20,      # 动量计算窗口
                 ma_short_window=5,       # 短期均线
                 ma_long_window=20,       # 长期均线
                 top_n_sectors=3,         # 选择的行业数量
                 initial_capital=1000000, # 初始资金
                 transaction_cost=0.002): # 交易成本
        
        self.data = pd.read_csv(data_path, encoding='GBK', parse_dates=['date'])
        self.momentum_window = momentum_window
        self.ma_short_window = ma_short_window
        self.ma_long_window = ma_long_window
        self.top_n_sectors = top_n_sectors
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def calculate_signals(self):
        """计算交易信号"""
        df = self.data.copy()
        
        # 按行业分组计算指标
        for industry in df['first_industry_name'].unique():
            mask = df['first_industry_name'] == industry
            
            # 1. 计算动量因子（过去N日收益率）
            df.loc[mask, 'momentum'] = df.loc[mask, 'index_return'].rolling(
                self.momentum_window).sum()
            
            # 2. 计算均线
            df.loc[mask, 'ma_short'] = df.loc[mask, 'close'].rolling(
                self.ma_short_window).mean()
            df.loc[mask, 'ma_long'] = df.loc[mask, 'close'].rolling(
                self.ma_long_window).mean()
            
            # 3. 计算趋势信号（短期均线是否大于长期均线）
            df.loc[mask, 'trend'] = (df.loc[mask, 'ma_short'] > 
                                   df.loc[mask, 'ma_long']).astype(int)
        
        return df
    
    def select_sectors(self, date_data):
        """选择最优行业"""
        # 确保数据有效
        valid_data = date_data.dropna(subset=['momentum', 'trend'])
        
        # 综合评分：动量 * 趋势信号
        valid_data['score'] = valid_data['momentum'] * valid_data['trend']
        
        # 选择得分最高的前N个行业
        selected = valid_data.nlargest(self.top_n_sectors, 'score')
        
        # 如果选中的行业数量不足，补充为现金
        if len(selected) < self.top_n_sectors:
            return selected
        
        return selected
    
    def backtest(self, start_date=None, end_date=None):
        """回测策略"""
        # 计算信号
        df = self.calculate_signals()
        
        # 设置回测区间
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]
        
        # 初始化回测变量
        cash = self.initial_capital
        positions = {}  # 持仓
        net_values = []  # 净值
        
        # 按日期遍历
        for date in sorted(df['date'].unique()):
            date_data = df[df['date'] == date]
            
            # 选择行业
            selected_sectors = self.select_sectors(date_data)
            
            if len(selected_sectors) > 0:
                # 计算每个行业的目标权重（等权重）
                weight_per_sector = 1.0 / self.top_n_sectors
                
                # 更新持仓
                new_positions = {}
                total_value = cash
                
                # 计算当前持仓市值
                for industry, shares in positions.items():
                    if industry in date_data['first_industry_name'].values:
                        price = date_data[date_data['first_industry_name'] == industry]['close'].iloc[0]
                        total_value += shares * price
                
                # 计算目标持仓
                for _, row in selected_sectors.iterrows():
                    target_value = total_value * weight_per_sector
                    shares = target_value / row['close']
                    new_positions[row['first_industry_name']] = shares
                
                # 计算交易成本
                cost = 0
                for industry, shares in positions.items():
                    if industry not in new_positions:
                        # 卖出成本
                        price = date_data[date_data['first_industry_name'] == industry]['close'].iloc[0]
                        cost += shares * price * self.transaction_cost
                
                for industry, shares in new_positions.items():
                    if industry not in positions:
                        # 买入成本
                        price = date_data[date_data['first_industry_name'] == industry]['close'].iloc[0]
                        cost += shares * price * self.transaction_cost
                
                # 更新现金和持仓
                cash = total_value - sum(shares * date_data[date_data['first_industry_name'] == industry]['close'].iloc[0] 
                                      for industry, shares in new_positions.items()) - cost
                positions = new_positions
            
            # 计算当日净值
            portfolio_value = cash
            for industry, shares in positions.items():
                if industry in date_data['first_industry_name'].values:
                    price = date_data[date_data['first_industry_name'] == industry]['close'].iloc[0]
                    portfolio_value += shares * price
            
            net_values.append({
                'date': date,
                'net_value': portfolio_value
            })
        
        return pd.DataFrame(net_values)
    
    def evaluate_strategy(self, net_values):
        """评估策略表现"""
        net_values['daily_returns'] = net_values['net_value'].pct_change()
        
        # 计算评估指标
        total_return = (net_values['net_value'].iloc[-1] / net_values['net_value'].iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(net_values)) - 1
        daily_std = net_values['daily_returns'].std()
        sharpe_ratio = np.sqrt(252) * net_values['daily_returns'].mean() / daily_std
        
        # 计算最大回撤
        net_values['rolling_max'] = net_values['net_value'].cummax()
        net_values['drawdown'] = net_values['net_value'] / net_values['rolling_max'] - 1
        max_drawdown = net_values['drawdown'].min()
        
        print("\n策略评估结果:")
        print(f"总收益率: {total_return:.2%}")
        print(f"年化收益率: {annual_return:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"最大回撤: {max_drawdown:.2%}")
        
        # 绘制净值曲线
        plt.figure(figsize=(12, 6))
        plt.plot(net_values['date'], net_values['net_value'], label='策略净值')
        plt.title('行业轮动策略净值曲线')
        plt.xlabel('日期')
        plt.ylabel('净值')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    # 创建策略实例
    strategy = SimpleSectorRotation(
        data_path='industry_index_with_close2.csv',
        momentum_window=20,
        ma_short_window=5,
        ma_long_window=20,
        top_n_sectors=3,
        initial_capital=1000000,
        transaction_cost=0.002
    )
    
    # 回测策略
    net_values = strategy.backtest(start_date='2020-01-01', end_date='2023-12-31')
    
    # 评估策略
    strategy.evaluate_strategy(net_values)

if __name__ == "__main__":
    main() 