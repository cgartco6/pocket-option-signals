import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

def create_features(df):
    """Create features for AI model"""
    # Price features
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    
    # Volume features
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_spike'] = (df['volume'] / df['volume_ma']).replace([np.inf, -np.inf], 0)
    
    # Technical features
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd_diff(df['close'])
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
    df['atr_pct'] = df['atr'] / df['close']
    
    # Target: 1 if next candle is green, 0 if red
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    return df.dropna()

# Load historical data (replace with your data file)
# Columns should include: timestamp, open, high, low, close, volume
historical_data = pd.read_csv('historical_data.csv', parse_dates=['timestamp'])
historical_data.set_index('timestamp', inplace=True)

# Create features
featured_data = create_features(historical_data)

# Prepare training data
X = featured_data[['rsi', 'macd', 'volume_spike', 'atr_pct', 'volatility']]
y = featured_data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2%}")

# Save model
dump(model, 'trading_model.joblib')
print("AI model saved to trading_model.joblib")
