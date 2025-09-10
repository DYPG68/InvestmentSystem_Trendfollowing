#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from tradingview_screener import Query, Column

class InvestmentSystem:
    def __init__(self, capital=10000):
        self.capital = capital  # 本金 HKD
        self.risk_per_trade = 100  # 每單風險 HKD (1%)
        self.max_total_risk = 5000  # 總風險上限 HKD (50%)
        self.max_stocks = 3  # 同時持有股票數
        self.max_investment_per_stock = 1250  # 單股投資上限 HKD (12.5%)
        self.current_investment = 0
        self.current_risk = 0
        self.exchange_rate = 7.8  # USD/HKD匯率
        self.transaction_cost_per_share = 0.0035  # IBKR Tiered Pricing: $0.0035/股 (≤ 300,000 股/月)
        self.minimum_transaction_cost = 0.35  # IBKR 最低手續費 $0.35
        self.portfolio_file = 'portfolio.json'
        self.watchlist_file = 'watchlist.json'
        self.load_portfolio()
        self.load_watchlist()

    def load_portfolio(self):
        try:
            with open(self.portfolio_file, 'r') as f:
                self.portfolio = json.load(f)
        except FileNotFoundError:
            self.portfolio = []
        self.update_current_metrics()

    def save_portfolio(self):
        with open(self.portfolio_file, 'w') as f:
            json.dump(self.portfolio, f, indent=2)

    def load_watchlist(self):
        try:
            with open(self.watchlist_file, 'r') as f:
                self.watchlist = json.load(f)
        except FileNotFoundError:
            self.watchlist = []

    def save_watchlist(self):
        with open(self.watchlist_file, 'w') as f:
            json.dump(self.watchlist, f, indent=2)

    def update_current_metrics(self):
        self.current_investment = sum(stock['investment_hkd'] for stock in self.portfolio)
        self.current_risk = sum(stock['risk_hkd'] for stock in self.portfolio)

    def check_market_conditions(self):
        """檢查大盤趨勢條件，若下跌則清倉"""
        print("📊 大盤趨勢檢查")
        print("=" * 50)
        try:
            spy_query = (Query()
                         .set_markets('america')
                         .select('name', 'close', 'EMA30|1W')
                         .where(Column('name') == 'SPY'))
            _, spy_df = spy_query.get_scanner_data()
            if not spy_df.empty:
                spy_data = spy_df.iloc[0]
                current_price = spy_data['close']
                weekly_ema30 = spy_data['EMA30|1W']
                market_healthy = current_price > weekly_ema30
                print(f"大盤代表: SPY (S&P 500 ETF)")
                print(f"當前價格: ${current_price:.2f}")
                print(f"週線EMA30: ${weekly_ema30:.2f}")
                print("-" * 40)
                if not market_healthy:
                    print("❌ 趨勢狀態: 📉 不健康")
                    print("   • 價格低於週線EMA30")
                    print("🚫 清倉所有持倉")
                    self.clear_portfolio()
                    return False
                else:
                    print("✅ 趨勢狀態: 📈 健康")
                    print("   • 價格在週線EMA30之上")
                    print("✅ 可以開始選股")
                    return True
            else:
                print("❌ 無法獲取SPY數據")
                print("⚠️ 假設使用週五收盤數據，繼續運行")
                return None
        except Exception as e:
            print(f"❌ 檢查失敗: {e}")
            print("⚠️ 假設使用週五收盤數據，繼續運行")
            return None

    def clear_portfolio(self):
        """清倉所有持倉"""
        if not self.portfolio:
            print("❌ 無持倉股票可清倉")
            return
        current_data_df = self.get_current_data([stock['symbol'] for stock in self.portfolio])
        current_data_map = {row['name']: row for _, row in current_data_df.iterrows()} if not current_data_df.empty else {}
        for stock in self.portfolio:
            symbol = stock['symbol']
            current_data = current_data_map.get(symbol, {'close': stock['current_price'], 'EMA30|1W': 0, 'price_52_week_high': stock['current_price']})
            print(f"💸 清倉 {symbol}")
            self.handle_sell(stock, current_data)
        self.portfolio = []
        self.save_portfolio()
        self.update_current_metrics()

    def screen_stocks(self):
        """自動選股程序，確保財報後且不包含已持倉股票"""
        print("\n🔍 開始自動選股...")
        print("=" * 50)
        query = (Query()
                 .set_markets('america')
                 .select(
                     'name', 'close', 'high', 'volume', 'relative_volume_10d_calc',
                     'EMA10', 'EMA20', 'EMA50', 'EMA100', 'ADX',
                     'High.3M', 'High.6M', 'price_52_week_high',
                     'average_volume_90d_calc', 'market_cap_basic', 'sector', 'industry',
                     'EMA30|1W', 'ATR|1W', 'earnings_release_date'
                 )
                 .where(
                     Column('market_cap_basic') > 500000000,  # 市值 > 0.5B USD
                     Column('close').between(20, 160),  # 價格區間 20-160 USD
                     Column('close') > Column('EMA30|1W'),  # 價格 > 30週EMA
                     Column('EMA50') > Column('EMA100'),  # 多頭排列
                     Column('relative_volume_10d_calc') > 1.5,  # 成交量 > 前10天平均1.5倍
                 ))
        try:
            _, df = query.get_scanner_data()
            if df.empty:
                print("❌ 沒有找到符合條件的股票")
                return pd.DataFrame()
            print(f"找到 {len(df)} 支符合基本條件的股票")
            # 排除已持倉股票
            current_holdings = [stock['symbol'] for stock in self.portfolio]
            if current_holdings:
                df = df[~df['name'].isin(current_holdings)]
                print(f"排除 {len(current_holdings)} 支已持倉股票，剩餘 {len(df)} 支股票")
            if df.empty:
                print("❌ 排除持倉後無符合條件的股票")
                return pd.DataFrame()
            # 處理 earnings_release_date（Unix 時間戳）
            print("\n📅 處理財報日期...")
            df['earnings_release_date'] = df['earnings_release_date'].map(
                lambda x: pd.Timestamp.utcfromtimestamp(x) if pd.notnull(x) else pd.NaT,
                na_action='ignore'
            )
            # 記錄無財報數據的股票
            invalid_earnings = df[df['earnings_release_date'].isna()]
            if not invalid_earnings.empty:
                print(f"⚠️ {len(invalid_earnings)} 支股票無財報日期數據：")
                for _, row in invalid_earnings[['name', 'sector']].iterrows():
                    print(f"   • {row['name']} ({row['sector']})")
            # 過濾最近 30 天的財報
            thirty_days_ago = pd.Timestamp.now(tz='UTC') - timedelta(days=30)
            df_valid = df[df['earnings_release_date'].notna()]
            df_filtered = df_valid[df_valid['earnings_release_date'] >= thirty_days_ago]
            if df_filtered.empty:
                print("❌ 無股票在最近30天發布財報")
                print("ℹ️ 已記錄無財報數據的股票，建議檢查 TradingView 網站")
                return pd.DataFrame()
            print(f"財報過濾後剩下 {len(df_filtered)} 支股票")
            # 後續過濾
            df_filtered = df_filtered[df_filtered['average_volume_90d_calc'] * df_filtered['close'] > 10000000]  # 日均成交額 > 10M USD
            df_filtered = df_filtered[np.isclose(df_filtered['high'], df_filtered['High.3M'])]  # 突破3個月高
            df_filtered = df_filtered[np.isclose(df_filtered['high'], df_filtered['High.6M'])]  # 突破6個月高
            df_filtered = df_filtered[np.isclose(df_filtered['high'], df_filtered['price_52_week_high'])]  # 突破52週高
            if df_filtered.empty:
                print("❌ 後續過濾後沒有符合突破訊號的股票")
                return pd.DataFrame()
            print(f"突破過濾後剩下 {len(df_filtered)} 支股票")
            df_filtered['trend_strength'] = self.calculate_trend_strength(df_filtered)
            best_by_sector = pd.DataFrame()
            for sector, group in df_filtered.groupby('sector'):
                if not group.empty:
                    best_stock = group.nlargest(1, 'trend_strength').iloc[0]
                    best_by_sector = pd.concat([best_by_sector, pd.DataFrame([best_stock])], ignore_index=True)
            final_selection = best_by_sector.nlargest(3, 'trend_strength')
            print(f"最終選出 {len(final_selection)} 支股票")
            # 輸出最終選股的財報日期
            print("\n📋 最終選股財報日期：")
            for _, row in final_selection[['name', 'earnings_release_date']].iterrows():
                print(f"   • {row['name']}: {row['earnings_release_date'].strftime('%Y-%m-%d')}")
            return final_selection
        except Exception as e:
            print(f"❌ 選股錯誤: {e}")
            return pd.DataFrame()

    def calculate_trend_strength(self, df):
        """計算趨勢強度評分"""
        ema_score = np.where(df['close'] > df['EMA10'], 10, 0)
        ema_score += np.where(df['EMA10'] > df['EMA20'], 10, 0)
        ema_score += np.where(df['EMA20'] > df['EMA50'], 10, 0)
        ema_score += np.where(df['EMA50'] > df['EMA100'], 10, 0)
        adx_score = np.minimum(30, df['ADX'])
        breakthrough_score = np.minimum(20, df['ADX'] / 2)
        volume_score = np.minimum(20, (df['relative_volume_10d_calc'] - 1.5) * 20)
        pressure_penalty = np.where(df['relative_volume_10d_calc'] < 2, -5, 0)
        return ema_score + adx_score + breakthrough_score + volume_score + pressure_penalty

    def calculate_position_size(self, entry_price, atr, is_add=False):
        """計算部位大小，固定1%風險"""
        max_investment_usd = min(self.max_investment_per_stock / self.exchange_rate,
                                 (self.capital - self.current_investment) / self.exchange_rate)
        if max_investment_usd <= 0:
            return 0, 0, 0
        # 考慮 IBKR Tiered Pricing 手續費 ($0.0035/股，最低 $0.35)
        max_shares = int(max_investment_usd / entry_price)  # 初步計算股數
        transaction_cost = max(self.transaction_cost_per_share * max_shares, self.minimum_transaction_cost)
        adjusted_max_investment_usd = max_investment_usd - transaction_cost
        if adjusted_max_investment_usd <= 0:
            return 0, 0, 0
        max_shares = int(adjusted_max_investment_usd / entry_price)  # 調整後股數
        # 假設初始止蝕距離為入場價的10%（可調整）
        risk_per_share = entry_price * 0.10
        risk_based_shares = int(self.risk_per_trade / (risk_per_share * self.exchange_rate))
        shares = min(max_shares, risk_based_shares)
        if shares <= 0:
            return 0, 0, 0
        investment_usd = shares * entry_price
        transaction_cost = max(self.transaction_cost_per_share * shares, self.minimum_transaction_cost)
        total_cost_hkd = (investment_usd + transaction_cost) * self.exchange_rate
        # 止蝕價確保固定1%風險
        stop_loss = entry_price - (self.risk_per_trade / (shares * self.exchange_rate))
        return shares, total_cost_hkd, stop_loss

    def buy_stocks(self, selected_stocks):
        """買入選定的股票"""
        signals = []
        for _, stock in selected_stocks.iterrows():
            if len(self.portfolio) >= self.max_stocks or self.current_risk >= self.max_total_risk:
                break
            symbol = stock['name']
            current_price = stock['close']
            atr = stock['ATR|1W']
            shares, total_cost, stop_loss = self.calculate_position_size(current_price, atr)
            if shares > 0:
                risk_amount = (current_price - stop_loss) * shares * self.exchange_rate
                self.portfolio.append({
                    'symbol': symbol,
                    'shares': shares,
                    'cost_price': current_price,
                    'current_price': current_price,
                    'stop_loss': stop_loss,
                    'atr': atr,
                    'investment_hkd': total_cost,
                    'risk_hkd': risk_amount,
                    'sector': stock['sector'],
                    'entry_date': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat(),
                    'trailing_stop': False
                })
                self.current_investment += total_cost
                self.current_risk += risk_amount
                signals.append({
                    'symbol': symbol,
                    'shares': shares,
                    'entry_price': current_price,
                    'stop_loss': stop_loss
                })
        self.save_portfolio()
        return signals

    def get_current_data(self, symbols):
        """獲取當前股票數據"""
        print(f"📡 獲取 {len(symbols)} 支股票當前數據...")
        try:
            query = (Query()
                     .set_markets('america')
                     .select(
                         'name', 'close', 'high', 'low', 'open', 'volume',
                         'EMA10', 'EMA20', 'EMA50', 'EMA30|1W', 'ATR|1W',
                         'sector', 'industry', 'change|1W', 'gap', 'price_52_week_high'
                     )
                     .where(Column('name').isin(symbols)))
            _, df = query.get_scanner_data()
            if df.empty:
                print("❌ 未獲取到數據，假設使用上次數據")
                return pd.DataFrame()
            print(f"✅ 成功獲取 {len(df)} 支股票數據")
            return df
        except Exception as e:
            print(f"❌ 獲取數據失敗: {e}")
            print("⚠️ 假設使用上次數據")
            return pd.DataFrame()

    def monitor_portfolio(self):
        """監控持倉股，更新追蹤止蝕"""
        print("\n📊 開始監控持倉...")
        print("=" * 50)
        if not self.portfolio:
            print("❌ 無持倉股票")
            return
        symbols = [stock['symbol'] for stock in self.portfolio]
        print(f"監控股票: {', '.join(symbols)}")
        current_data_df = self.get_current_data(symbols)
        if current_data_df.empty:
            print("❌ 無法獲取市場數據，使用上次數據")
            current_data_map = {stock['symbol']: {
                'name': stock['symbol'],
                'close': stock['current_price'],
                'ATR|1W': stock['atr'],
                'EMA30|1W': 0,
                'price_52_week_high': stock['current_price'],
                'gap': 0
            } for stock in self.portfolio}
        else:
            current_data_map = {row['name']: row for _, row in current_data_df.iterrows()}
        updated_portfolio = []
        trailing_stop_updates = []
        for stock in self.portfolio:
            symbol = stock['symbol']
            if symbol not in current_data_map:
                print(f"❌ 無法獲取 {symbol} 的數據，保留上次數據")
                updated_portfolio.append(stock)
                continue
            current_data = current_data_map[symbol]
            current_price = current_data['close']
            cost_price = stock['cost_price']
            current_stop_loss = stock['stop_loss']
            atr = current_data['ATR|1W']
            price_change_pct = ((current_price - cost_price) / cost_price) * 100
            print(f"\n🔍 檢查 {symbol}:")
            print(f"   • 當前價格: ${current_price:.2f}")
            print(f"   • 成本價格: ${cost_price:.2f}")
            print(f"   • 當前止損: ${current_stop_loss:.2f}")
            print(f"   • ATR: ${atr:.2f}")
            print(f"   • 盈虧: {price_change_pct:+.1f}%")
            # 追蹤止蝕邏輯
            new_stop_loss = current_price - 2 * atr
            if new_stop_loss > cost_price and not stock.get('trailing_stop', False):
                stock['trailing_stop'] = True
                stock['stop_loss'] = new_stop_loss
                print(f"🛡️ {symbol}: 啟動追蹤止損，止損價 ${new_stop_loss:.2f}")
                trailing_stop_updates.append({'symbol': symbol, 'stop_loss': new_stop_loss, 'atr': atr})
            elif stock['trailing_stop']:
                if new_stop_loss > current_stop_loss:
                    stock['stop_loss'] = new_stop_loss
                    print(f"🛡️ {symbol}: 更新追蹤止蝕，止損價 ${new_stop_loss:.2f}")
                    trailing_stop_updates.append({'symbol': symbol, 'stop_loss': new_stop_loss, 'atr': atr})
                else:
                    print(f"🛡️ {symbol}: 價格下跌，維持止損價 ${current_stop_loss:.2f}")
                    trailing_stop_updates.append({'symbol': symbol, 'stop_loss': current_stop_loss, 'atr': atr})
            # 檢查是否觸及止損
            if current_price <= current_stop_loss:
                print(f"💸 平倉 {symbol}: 觸及止損")
                self.handle_sell(stock, current_data)
                continue
            stock['current_price'] = current_price
            stock['atr'] = atr
            stock['last_updated'] = datetime.now().isoformat()
            updated_portfolio.append(stock)
        self.portfolio = updated_portfolio
        self.save_portfolio()
        self.update_current_metrics()
        # 輸出追蹤止蝕清單
        if trailing_stop_updates:
            print("\n📋 週一止蝕設定清單（請於週一開盤前設定）:")
            for update in trailing_stop_updates:
                print(f"   • {update['symbol']}: 止損價 ${update['stop_loss']:.2f} (動態追蹤: 2 * ATR = ${2 * update['atr']:.2f})")
        print("\n✅ 監控完成")

    def add_position(self, stock, current_price):
        """執行加注操作"""
        if not stock['trailing_stop']:
            print("❌ 未滿足加注前提（追蹤止損）")
            return
        shares, total_cost, _ = self.calculate_position_size(current_price, stock['atr'], is_add=True)
        if shares <= 0:
            return
        total_shares = stock['shares'] + shares
        total_investment = stock['investment_hkd'] + total_cost
        new_cost_price = (stock['cost_price'] * stock['shares'] + current_price * shares) / total_shares
        stock['shares'] = total_shares
        stock['cost_price'] = new_cost_price
        stock['investment_hkd'] = total_investment
        stock['stop_loss'] = current_price - 2 * stock['atr']
        self.current_investment += total_cost
        self.current_risk += self.risk_per_trade
        print(f"➕ 加注 {stock['symbol']}: {shares} 股，新成本價 ${new_cost_price:.2f}")

    def handle_sell(self, stock, current_data):
        """處理平倉後邏輯"""
        symbol = stock['symbol']
        weekly_ema30 = current_data.get('EMA30|1W', 0)
        current_price = current_data['close']
        stronger_stocks = self.screen_stocks()
        has_stronger = not stronger_stocks.empty
        if current_price > weekly_ema30 and not has_stronger:
            self.watchlist.append({
                'symbol': symbol,
                'previous_high': current_data.get('price_52_week_high', current_price),
                'breakout_level': current_price * 1.05,
                'added_date': datetime.now().isoformat()
            })
            self.save_watchlist()
            print(f"👀 {symbol} 放入觀察區")
        else:
            print(f"🔄 {symbol} 釋放資金，尋找新股票")
        self.current_investment -= stock['investment_hkd']
        self.current_risk -= stock['risk_hkd']

    def check_watchlist(self):
        """檢查觀察區股票是否觸發再入場"""
        if not self.watchlist:
            print("❌ 無觀察區股票")
            return
        print("\n👀 檢查觀察區...")
        print("=" * 50)
        symbols = [w['symbol'] for w in self.watchlist]
        print(f"觀察股票: {', '.join(symbols)}")
        current_data_df = self.get_current_data(symbols)
        if current_data_df.empty:
            print("❌ 無法獲取數據，跳過")
            return
        current_data_map = {row['name']: row for _, row in current_data_df.iterrows()}
        updated_watchlist = []
        for w in self.watchlist:
            symbol = w['symbol']
            if symbol not in current_data_map:
                updated_watchlist.append(w)
                continue
            current_data = current_data_map[symbol]
            current_price = current_data['close']
            if current_price >= w['breakout_level'] or current_price > w['previous_high']:
                print(f"🎯 {symbol}: 觸發再入場訊號")
                selected = pd.DataFrame([current_data])
                self.buy_stocks(selected)
            else:
                print(f"   • {symbol}: 未觸發訊號，繼續觀察")
                updated_watchlist.append(w)
        self.watchlist = updated_watchlist
        self.save_watchlist()
        print("✅ 觀察區檢查完成")

    def run(self):
        """執行整個投資系統"""
        print("🚀 啟動投資系統")
        print("=" * 60)
        print("⚠️ 運行於週末，使用週五收盤數據，週一開盤後請檢查跳空")
        market_condition = self.check_market_conditions()
        if market_condition is False:
            print("🛑 暫停交易")
            return
        elif market_condition is None:
            print("⚠️ 無法檢查大盤，使用週五數據，繼續但請謹慎")
        # 監控現有持倉
        self.monitor_portfolio()
        # 檢查觀察區
        self.check_watchlist()
        # 如果有空間，選股買入
        if len(self.portfolio) < self.max_stocks and self.current_risk < self.max_total_risk:
            selected_stocks = self.screen_stocks()
            if not selected_stocks.empty:
                signals = self.buy_stocks(selected_stocks)
                if signals:
                    print("\n🎯 週一買入清單（請於週一開盤後一小時執行）:")
                    for sig in signals:
                        print(f"   • {sig['symbol']}: {sig['shares']} 股 @ ${sig['entry_price']:.2f}, 止損 ${sig['stop_loss']:.2f}")
                    print("⚠️ 請確認週一開盤價，若差距過大，重新計算股數和止損")
        # 顯示概要
        print("\n📊 系統概要:")
        print(f"• 總投資: HKD {self.current_investment:,.0f}")
        print(f"• 總風險: HKD {self.current_risk:,.0f}")
        print(f"• 持倉股票: {len(self.portfolio)}")
        print(f"• 觀察區股票: {len(self.watchlist)}")

if __name__ == "__main__":
    system = InvestmentSystem(capital=10000)
    system.run()


# In[ ]:




