# MVP Trading Signal System

Simple trading signal generator using CGARTCO moving average strategy with Telegram notifications.

## Features
- 5-period and 21-period EMA crossover strategy
- Binance API for market data
- Telegram notifications
- 15-minute signal cooldown

## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Configure:
   - `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `main.py`
   - Trading pairs in `ASSETS` (use Binance symbols)
3. Run: `python main.py`

## Trading Strategy
- **Buy Signal:** 5 EMA crosses above 21 EMA
- **Sell Signal:** 5 EMA crosses below 21 EMA
