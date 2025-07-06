# Advanced Trading Signal System

Professional trading system with multiple signal generators and risk management for Pocket Option.

## Features
- Three signal generation strategies:
  1. CGARTCO Moving Average (5 EMA & 21 EMA)
  2. Pocket Option proprietary signals
  3. AI Agent with 92%+ confidence
- 92%+ payout validation
- Risk management system
- Telegram trade alerts
- Position sizing based on account risk

## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Configure:
   - Telegram credentials in `main.py`
   - Pocket Option API key
   - Trading assets
3. (Optional) Train AI model:
   - Place historical data in `historical_data.csv`
   - Run `python train_model.py`
4. Run: `python main.py`

## Trading Strategies
- **CGARTCO MA Cross:** 5 EMA vs 21 EMA crossover
- **PO Strategy:** EMA20 + RSI + Volume confirmation
- **AI Agent:** Machine learning model with feature engineering
