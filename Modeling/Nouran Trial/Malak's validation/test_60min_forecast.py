"""
================================================================================
60-MINUTE AHEAD BGL FORECASTING TEST SCRIPT
================================================================================
Tests your models on fresh synthetic 30-minute data with HORIZON=2.

PREDICTION HORIZON: 60 minutes ahead (HORIZON=2)
INPUT WINDOW: 12 hours (WINDOW_SIZE=24, 30-min intervals)

NOTE: Your models were trained with HORIZON=1 (30-min). To test 60-min ahead,
you need to either:
  A) Retrain models with HORIZON=2, OR
  B) Use the 30-min model recursively (predict 30-min, then predict again)

This script implements Option B (recursive multi-step) as a demonstration.
For true 60-min forecasting, retrain with HORIZON=2.

REQUIREMENTS:
    pip install pandas numpy scikit-learn joblib matplotlib tensorflow

USAGE:
    python test_60min_forecast.py
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SYNTHETIC_DATA_PATH = 'synthetic_bgl_30min_test_data.csv'
MEDICAL_MIN = 40
MEDICAL_MAX = 400
WINDOW_SIZE = 24      # 12 hours of history
HORIZON_30MIN = 1     # Model was trained for 30-min ahead
HORIZON_60MIN = 2     # We want 60-min ahead

LSTM_MODEL_PATH = 'lstm_model.pkl'
GRU_MODEL_PATH = 'gru_model.pkl'
DENSE_MODEL_PATH = 'dense_model.pkl'
SCALER_PATH = 'scaler.pkl'

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("=" * 70)
print("  60-MINUTE AHEAD FORECASTING TEST")
print("=" * 70)

df = pd.read_csv(SYNTHETIC_DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

df_ts = df.set_index('date')[['bgl']].copy()
df_ts['bgl'] = df_ts['bgl'].clip(lower=MEDICAL_MIN, upper=MEDICAL_MAX)

if __import__('os').path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    print(f"  ✓ Loaded scaler")
else:
    scaler = MinMaxScaler()
    print("  ⚠️  Fitting new scaler")

synthetic_scaled = scaler.fit_transform(df_ts[['bgl']])

# ==============================================================================
# 2. CREATE SEQUENCES FOR 60-MIN AHEAD (Ground Truth)
# ==============================================================================
print("\n" + "=" * 70)
print("  CREATING 60-MIN AHEAD GROUND TRUTH")
print("=" * 70)

def create_sequences(data, window_size, horizon):
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size+horizon-1])
    return np.array(X), np.array(y)

# Ground truth for 60-min ahead
X_60, y_60_true = create_sequences(synthetic_scaled, WINDOW_SIZE, HORIZON_60MIN)
print(f"  60-min sequences: {len(X_60):,}")
print(f"  X shape: {X_60.shape}")
print(f"  y shape: {y_60_true.shape}")

# Also create 30-min sequences for recursive prediction
X_30, y_30_true = create_sequences(synthetic_scaled, WINDOW_SIZE, HORIZON_30MIN)

# ==============================================================================
# 3. RECURSIVE 60-MIN PREDICTION (Option B)
# ==============================================================================
print("\n" + "=" * 70)
print("  RECURSIVE 60-MIN FORECAST (30-min model x2)")
print("=" * 70)

def recursive_60min_predict(model, X_30_data, scaler, n_steps=2):
    """
    Predict 60-min ahead by applying 30-min model twice.
    Step 1: Predict t+30 from X(t)
    Step 2: Predict t+60 from X(t+1) where last value is prediction from step 1
    """
    predictions = []

    for i in range(len(X_30_data) - n_steps + 1):
        # Current window
        window = X_30_data[i].copy()

        # Step 1: Predict 30-min ahead
        pred_30_scaled = model.predict(window.reshape(1, WINDOW_SIZE, 1), verbose=0)

        # Step 2: Build new window with prediction appended
        window_step2 = np.roll(window, -1, axis=0)
        window_step2[-1] = pred_30_scaled[0, 0]

        # Predict 30-min ahead again = 60-min total
        pred_60_scaled = model.predict(window_step2.reshape(1, WINDOW_SIZE, 1), verbose=0)
        predictions.append(pred_60_scaled[0, 0])

    return np.array(predictions).reshape(-1, 1)

# Test LSTM recursive
print("\n  Testing LSTM recursive 60-min...")
try:
    from tensorflow.keras.models import load_model
    lstm_model = load_model(LSTM_MODEL_PATH)

    lstm_60_scaled = recursive_60min_predict(lstm_model, X_30, scaler)
    lstm_60 = scaler.inverse_transform(lstm_60_scaled)
    y_60_actual = scaler.inverse_transform(y_60_true[:len(lstm_60)])

    lstm_mae = mean_absolute_error(y_60_actual, lstm_60)
    lstm_rmse = np.sqrt(mean_squared_error(y_60_actual, lstm_60))
    lstm_mape = np.mean(np.abs((y_60_actual - lstm_60) / y_60_actual)) * 100

    print(f"  LSTM 60-min MAE:  {lstm_mae:.2f} mg/dL")
    print(f"  LSTM 60-min RMSE: {lstm_rmse:.2f} mg/dL")
    print(f"  LSTM 60-min MAPE: {lstm_mape:.2f}%")
    print(f"  LSTM 60-min Acc:  {100-lstm_mape:.2f}%")

    lstm_60_results = {'pred': lstm_60.flatten(), 'actual': y_60_actual.flatten(),
                       'mae': lstm_mae, 'rmse': lstm_rmse, 'mape': lstm_mape}
except Exception as e:
    print(f"  ✗ Failed: {e}")
    lstm_60_results = None

# Test GRU recursive
print("\n  Testing GRU recursive 60-min...")
try:
    gru_model = load_model(GRU_MODEL_PATH)

    gru_60_scaled = recursive_60min_predict(gru_model, X_30, scaler)
    gru_60 = scaler.inverse_transform(gru_60_scaled)
    y_60_actual_gru = scaler.inverse_transform(y_60_true[:len(gru_60)])

    gru_mae = mean_absolute_error(y_60_actual_gru, gru_60)
    gru_rmse = np.sqrt(mean_squared_error(y_60_actual_gru, gru_60))
    gru_mape = np.mean(np.abs((y_60_actual_gru - gru_60) / y_60_actual_gru)) * 100

    print(f"  GRU 60-min MAE:  {gru_mae:.2f} mg/dL")
    print(f"  GRU 60-min RMSE: {gru_rmse:.2f} mg/dL")
    print(f"  GRU 60-min MAPE: {gru_mape:.2f}%")
    print(f"  GRU 60-min Acc:  {100-gru_mape:.2f}%")

    gru_60_results = {'pred': gru_60.flatten(), 'actual': y_60_actual_gru.flatten(),
                      'mae': gru_mae, 'rmse': gru_rmse, 'mape': gru_mape}
except Exception as e:
    print(f"  ✗ Failed: {e}")
    gru_60_results = None

# ==============================================================================
# 4. NAIVE BASELINE (for comparison)
# ==============================================================================
print("\n" + "=" * 70)
print("  NAIVE BASELINE (persistence model)")
print("=" * 70)

# Naive: predict t+60 = current value (t)
naive_pred = scaler.inverse_transform(X_60[:, -1, :])  # Last value in window
naive_actual = scaler.inverse_transform(y_60_true)
naive_mae = mean_absolute_error(naive_actual, naive_pred)
naive_rmse = np.sqrt(mean_squared_error(naive_actual, naive_pred))
naive_mape = np.mean(np.abs((naive_actual - naive_pred) / naive_actual)) * 100

print(f"  Naive MAE:  {naive_mae:.2f} mg/dL")
print(f"  Naive RMSE: {naive_rmse:.2f} mg/dL")
print(f"  Naive MAPE: {naive_mape:.2f}%")
print(f"  Naive Acc:  {100-naive_mape:.2f}%")
print(f"  (Baseline: 'BGL in 60 min = BGL right now')")

# ==============================================================================
# 5. SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("  60-MINUTE AHEAD FORECAST SUMMARY")
print("=" * 70)

print(f"\n  {'Model':<15} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'Accuracy':>10}")
print(f"  {'-'*65}")
print(f"  {'Naive':<15} {naive_mae:>10.2f} {naive_rmse:>10.2f} {naive_mape:>9.2f}% {100-naive_mape:>9.2f}%")

if lstm_60_results:
    print(f"  {'LSTM (rec)':<15} {lstm_60_results['mae']:>10.2f} {lstm_60_results['rmse']:>10.2f} {lstm_60_results['mape']:>9.2f}% {100-lstm_60_results['mape']:>9.2f}%")
if gru_60_results:
    print(f"  {'GRU (rec)':<15} {gru_60_results['mae']:>10.2f} {gru_60_results['rmse']:>10.2f} {gru_60_results['mape']:>9.2f}% {100-gru_60_results['mape']:>9.2f}%")

print("\n  RECOMMENDATION:")
print("  For true 60-min forecasting, retrain models with HORIZON=2")
print("  Recursive prediction accumulates error at each step.")

# ==============================================================================
# 6. PLOT
# ==============================================================================
if lstm_60_results or gru_60_results:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if lstm_60_results:
        axes[0].plot(lstm_60_results['actual'][:200], label='Actual', color='black', linewidth=1.5)
        axes[0].plot(lstm_60_results['pred'][:200], label='LSTM 60-min', color='blue', linewidth=1.5, alpha=0.8)
        axes[0].set_title(f'LSTM Recursive 60-min (MAE={lstm_60_results["mae"]:.1f})')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('BGL (mg/dL)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    if gru_60_results:
        axes[1].plot(gru_60_results['actual'][:200], label='Actual', color='black', linewidth=1.5)
        axes[1].plot(gru_60_results['pred'][:200], label='GRU 60-min', color='green', linewidth=1.5, alpha=0.8)
        axes[1].set_title(f'GRU Recursive 60-min (MAE={gru_60_results["mae"]:.1f})')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('BGL (mg/dL)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.suptitle('60-Minute Ahead BGL Forecasting', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('forecast_60min_results.png', dpi=150, bbox_inches='tight')
    print("\n  ✓ Saved: forecast_60min_results.png")
    plt.show()

print("\n" + "=" * 70)
print("  TEST COMPLETE")
print("=" * 70)
