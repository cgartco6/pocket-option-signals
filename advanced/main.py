import pandas as pd
import numpy as np
import ta
import requests
from datetime import datetime, timedelta
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from joblib import load
import os

# Configuration
TELEGRAM_BOT_TOKEN = '7475784679:AAFK9Y183wxB_5YriDvkflYzveNwZRAt9vE'
TELEGRAM_CHAT_ID = '7642813067'
POCKET_OPTION_API_KEY = 'AKkBM7jXxdzaFsyj5'
ASSETS = ['EURUSD', 'GBPUSD', 'BTCUSD', 'ETHUSD', 'USDJPY']  # Use without slashes
TIMEFRAME = '5m'
MIN_PAYOUT = 92
MIN_CONFIDENCE = 0.92
BALANCE = 10000
RISK_PERCENT = 1

# Asset-specific settings
ASSET_SETTINGS = {
    'EURUSD': {'expiration': 5, 'stop_pct': 1.0, 'take_profit_pct': 2.0},
    'GBPUSD': {'expiration': 5, 'stop_pct': 1.0, 'take_profit_pct': 2.0},
    'BTCUSD': {'expiration': 3, 'stop_pct': 1.5, 'take_profit_pct': 3.0},
    'ETHUSD': {'expiration': 3, 'stop_pct': 1.5, 'take_profit_pct': 3.0},
    'USDJPY': {'expiration': 5, 'stop_pct': 1.0, 'take_profit_pct': 2.0},
}

class AdvancedTradingSystem:
    def __init__(self):
        self.data_cache = {}
        self.last_signal = {}
        self.account_balance = BALANCE
        self.ai_model = self.load_ai_model()
        self.scaler = RobustScaler()
        
    def load_ai_model(self):
        """Load trained AI model if available"""
        model_path = 'trading_model.joblib'
        if os.path.exists(model_path):
            try:
                return load(model_path)
            except:
                print("Error loading AI model")
        return None
        
    def fetch_data(self, asset):
        """Fetch market data from best available source"""
        # First try Pocket Option API
        po_data = self.fetch_pocket_option_data(asset)
        if po_data is not None:
            return po_data
            
        # Fallback to Binance
        return self.fetch_binance_data(asset)
    
    def fetch_pocket_option_data(self, asset):
        """Fetch data from Pocket Option API with payout validation"""
        url = f"https://api.pocketoption.com/chart/history?symbol={asset}&resolution={TIMEFRAME}"
        headers = {'Authorization': f'Bearer {POCKET_OPTION_API_KEY}'}
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                payout = data.get('payout', 0)
                if payout < MIN_PAYOUT:
                    return None
                    
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(data['t'], unit='s'),
                    'open': data['o'],
                    'high': data['h'],
                    'low': data['l'],
                    'close': data['c'],
                    'volume': data['v']
                }).set_index('timestamp')
                df['payout'] = payout
                return df
        except:
            return None
            
    def fetch_binance_data(self, asset):
        """Fetch data from Binance API"""
        url = f"https://api.binance.com/api/v3/klines?symbol={asset}&interval={TIMEFRAME}&limit=100"
        try:
            response = requests.get(url, timeout=5)
            data = response.json()
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_quote_volume', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            df.set_index('timestamp', inplace=True)
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            df['payout'] = MIN_PAYOUT  # Assume minimum payout
            return df
        except:
            return None

    def generate_signals(self, asset, df):
        """Generate all trading signals"""
        signals = []
        
        # Add signals from different strategies
        signals.extend(self.generate_cgartco_signals(df))
        signals.extend(self.generate_po_signals(df))
        signals.extend(self.generate_ai_signals(df))
        
        return signals

    def generate_cgartco_signals(self, df):
        """CGARTCO Moving Average Strategy"""
        if df is None or len(df) < 25:
            return []
            
        df['ema5'] = ta.trend.ema_indicator(df['close'], window=5)
        df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        # Golden Cross
        if current['ema5'] > current['ema21'] and prev['ema5'] <= prev['ema21']:
            signals.append(('BUY', 'CGARTCO Golden Cross'))
        # Death Cross
        elif current['ema5'] < current['ema21'] and prev['ema5'] >= prev['ema21']:
            signals.append(('SELL', 'CGARTCO Death Cross'))
            
        return signals

    def generate_po_signals(self, df):
        """Pocket Option proprietary signals"""
        if df is None or len(df) < 50:
            return []
            
        df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['volume_ma'] = df['volume'].rolling(5).mean()
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        if (current['close'] > current['ema20'] and 
            current['rsi'] > 55 and
            current['close'] > prev['high'] and
            current['volume'] > current['volume_ma']):
            signals.append(('BUY', 'PO Strategy'))
        elif (current['close'] < current['ema20'] and 
              current['rsi'] < 45 and 
              current['close'] < prev['low'] and
              current['volume'] > current['volume_ma']):
            signals.append(('SELL', 'PO Strategy'))
            
        return signals

    def generate_ai_signals(self, df):
        """AI-generated signals"""
        if df is None or len(df) < 100 or self.ai_model is None:
            return []
            
        # Feature engineering
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_spike'] = df['volume'] / df['volume_ma']
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
        df['atr_pct'] = df['atr'] / df['close']
        
        # Prepare data
        features = df[['rsi', 'macd', 'volume_spike', 'atr_pct']].iloc[-1:].values
        
        # Generate prediction
        try:
            features = self.scaler.transform(features)
            proba = self.ai_model.predict_proba(features)[0]
            confidence = max(proba)
            prediction = self.ai_model.predict(features)[0]
            
            if confidence >= MIN_CONFIDENCE:
                signal = 'BUY' if prediction == 1 else 'SELL'
                return [(signal, f'AI Agent ({confidence:.2%})')]
        except:
            pass
            
        return []

    def calculate_position_size(self, entry_price, stop_price):
        """Calculate position size based on risk management"""
        risk_amount = self.account_balance * (RISK_PERCENT / 100)
        risk_per_share = abs(entry_price - stop_price)
        return risk_amount / risk_per_share

    def generate_trade_params(self, asset, direction, current_price):
        """Generate trade parameters"""
        settings = ASSET_SETTINGS.get(asset, {'expiration': 5, 'stop_pct': 1.5, 'take_profit_pct': 3.0})
        
        if direction == 'BUY':
            stop_loss = current_price * (1 - settings['stop_pct']/100)
            take_profit = current_price * (1 + settings['take_profit_pct']/100)
        else:
            stop_loss = current_price * (1 + settings['stop_pct']/100)
            take_profit = current_price * (1 - settings['take_profit_pct']/100)
            
        position_size = self.calculate_position_size(current_price, stop_loss)
            
        return {
            'expiration': settings['expiration'],
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'risk_reward': settings['take_profit_pct'] / settings['stop_pct']
        }

    def send_telegram_alert(self, asset, signal, source, price, payout, params):
        """Send detailed trade alert to Telegram"""
        message = (
            f"🚀 **HIGH PAYOUT SIGNAL** 🚀\n"
            f"**Asset**: {asset}\n"
            f"**Signal**: {signal} ({source})\n"
            f"**Payout**: {payout}%\n"
            f"**Entry Price**: ${price:.5f}\n"
            f"**Stop Loss**: ${params['stop_loss']:.5f}\n"
            f"**Take Profit**: ${params['take_profit']:.5f}\n"
            f"**Expiration**: {params['expiration']} minutes\n"
            f"**Position Size**: {params['position_size']:.2f} units\n"
            f"**Risk/Reward**: 1:{params['risk_reward']:.1f}\n"
            f"**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"*Risk {RISK_PERCENT}% of account per trade*"
        )
        
        url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }
        try:
            requests.post(url, json=payload)
            print(f"Sent Telegram alert for {asset} {signal}")
        except Exception as e:
            print(f"Telegram error: {str(e)}")

    def is_new_signal(self, asset, signal_type, direction):
        """Prevent duplicate signals"""
        key = f"{asset}-{signal_type}-{direction}"
        if key in self.last_signal:
            if datetime.now() - self.last_signal[key] < timedelta(minutes=15):
                return False
        self.last_signal[key] = datetime.now()
        return True

    def run(self):
        """Main trading loop"""
        print("Advanced Trading System Started")
        while True:
            for asset in ASSETS:
                try:
                    df = self.fetch_data(asset)
                    if df is None or df.empty:
                        continue
                        
                    current_price = df['close'].iloc[-1]
                    payout = df['payout'].iloc[-1]
                    
                    # Skip if payout too low
                    if payout < MIN_PAYOUT:
                        continue
                        
                    signals = self.generate_signals(asset, df)
                    
                    for direction, source in signals:
                        if not self.is_new_signal(asset, source, direction):
                            continue
                            
                        params = self.generate_trade_params(asset, direction, current_price)
                        self.send_telegram_alert(asset, direction, source, current_price, payout, params)
                        print(f"Signal generated: {asset} {direction} ({source})")
                        
                except Exception as e:
                    print(f"Error processing {asset}: {str(e)}")
                    
            # Wait before next run
            time.sleep(60)

if __name__ == "__main__":
    system = AdvancedTradingSystem()
    system.run()
