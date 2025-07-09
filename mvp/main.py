import pandas as pd
import ta
import requests
import time
from datetime import datetime

# Configuration
TELEGRAM_BOT_TOKEN = '7928426674:AAElIFCM4hXJQNFvUKPcJFbmika2gohIAIc'
TELEGRAM_CHAT_ID = '-1002672363651'
ASSETS = ['EURUSD', 'BTCUSDT']  # Use trading pairs without slashes
TIMEFRAME = '5m'
LIMIT = 50  # Number of candles to fetch

class MVPTradingSystem:
    def __init__(self):
        self.last_signal = {}
        
    def fetch_data(self, asset):
        """Fetch market data from Binance API"""
        url = f"https://api.binance.com/api/v3/klines?symbol={asset}&interval={TIMEFRAME}&limit={LIMIT}"
        try:
            response = requests.get(url)
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
                
            return df
        except Exception as e:
            print(f"Error fetching data for {asset}: {str(e)}")
            return None

    def generate_signals(self, df):
        """CGARTCO Moving Average Strategy"""
        if df is None or len(df) < 25:
            return []
            
        # Calculate EMAs
        df['ema5'] = ta.trend.ema_indicator(df['close'], window=5)
        df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        # Golden Cross
        if current['ema5'] > current['ema21'] and prev['ema5'] <= prev['ema21']:
            signals.append('BUY')
        # Death Cross
        elif current['ema5'] < current['ema21'] and prev['ema5'] >= prev['ema21']:
            signals.append('SELL')
            
        return signals

    def send_telegram_alert(self, asset, signal):
        """Send signal notification to Telegram"""
        message = (
            f"🚨 TRADE SIGNAL 🚨\n"
            f"Asset: {asset}\n"
            f"Signal: {signal}\n"
            f"Strategy: CGARTCO MA Cross\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message
        }
        try:
            requests.post(url, json=payload)
            print(f"Telegram alert sent for {asset} {signal}")
        except Exception as e:
            print(f"Failed to send Telegram alert: {str(e)}")

    def run(self):
        """Main trading loop"""
        print("MVP Trading System Started")
        while True:
            for asset in ASSETS:
                try:
                    df = self.fetch_data(asset)
                    signals = self.generate_signals(df)
                    
                    for signal in signals:
                        # Prevent duplicate signals
                        last_signal_key = f"{asset}-{signal}"
                        last_signal_time = self.last_signal.get(last_signal_key, datetime.min)
                        
                        # 15-minute cooldown
                        if (datetime.now() - last_signal_time).total_seconds() < 900:
                            continue
                            
                        self.last_signal[last_signal_key] = datetime.now()
                        self.send_telegram_alert(asset, signal)
                        print(f"Signal generated: {asset} {signal}")
                        
                except Exception as e:
                    print(f"Error processing {asset}: {str(e)}")
                    
            # Wait before next check
            time.sleep(60)

if __name__ == "__main__":
    trading_system = MVPTradingSystem()
    trading_system.run()
