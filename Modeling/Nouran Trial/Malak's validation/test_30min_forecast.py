"""
================================================================================
30-MINUTE AHEAD BGL FORECASTING TEST SCRIPT
================================================================================
Tests your models on fresh synthetic 30-minute data.

PREDICTION HORIZON: 30 minutes ahead (HORIZON=1)
INPUT WINDOW: 12 hours (WINDOW_SIZE=24, 30-min intervals)

Models tested:
  - LSTM  (true forecaster)
  - GRU   (true forecaster)
  - Dense (true forecaster)
  - HMM   (state estimator - shown for comparison)

REQUIREMENTS:
    pip install pandas numpy scikit-learn joblib matplotlib tensorflow

USAGE:
    python test_30min_forecast.py
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
WINDOW_SIZE = 24      # 12 hours of history (24 x 30-min)
HORIZON = 1           # 30 minutes ahead

# Model files (update these paths to match your saved models)
LSTM_MODEL_PATH = 'lstm_model.pkl'      # or .h5 / .keras
GRU_MODEL_PATH = 'gru_model.pkl'        # or .h5 / .keras
DENSE_MODEL_PATH = 'dense_model.pkl'    # or .h5 / .keras
HMM_MODEL_PATH = 'hmm_model.pkl'
SCALER_PATH = 'scaler.pkl'              # OPTIONAL: save during training!

# ==============================================================================
# 1. LOAD SYNTHETIC 30-MIN DATA
# ==============================================================================
print("=" * 70)
print("  LOADING FRESH SYNTHETIC 30-MIN DATA")
print("=" * 70)

df = pd.read_csv(SYNTHETIC_DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"\nSynthetic 30-min data loaded:")
print(f"  Records: {len(df):,}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
print(f"  Duration: {(df['date'].max() - df['date'].min()).days} days")
print(f"  BGL mean: {df['bgl'].mean():.2f} mg/dL")
print(f"  BGL std:  {df['bgl'].std():.2f} mg/dL")

# ==============================================================================
# 2. PREPROCESS (matching your new notebook pipeline)
# ==============================================================================
print("\n" + "=" * 70)
print("  PREPROCESSING (30-min resampled pipeline)")
print("=" * 70)

df_ts = df.set_index('date')[['bgl']].copy()

# Clip to medical bounds
df_ts['bgl'] = df_ts['bgl'].clip(lower=MEDICAL_MIN, upper=MEDICAL_MAX)

# Scale
if __import__('os').path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    print(f"  ✓ Loaded scaler from {SCALER_PATH}")
else:
    scaler = MinMaxScaler()
    print("  ⚠️  Fitting new scaler on synthetic data (ideally use training scaler)")

synthetic_scaled = scaler.fit_transform(df_ts[['bgl']])
print(f"  Scaler range: [{scaler.data_min_[0]:.2f}, {scaler.data_max_[0]:.2f}]")

# ==============================================================================
# 3. CREATE SEQUENCES FOR TRUE FORECASTING
# ==============================================================================
print("\n" + "=" * 70)
print("  CREATING FORECASTING SEQUENCES")
print("=" * 70)

def create_sequences(data, window_size, horizon=1):
    """Create X/y pairs for multi-step ahead forecasting."""
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size+horizon-1])
    return np.array(X), np.array(y)

X_synth, y_synth = create_sequences(synthetic_scaled, WINDOW_SIZE, horizon=HORIZON)
print(f"  Window size: {WINDOW_SIZE} steps = {WINDOW_SIZE*30} minutes = {WINDOW_SIZE*30/60} hours")
print(f"  Horizon: {HORIZON} step = {HORIZON*30} minutes ahead")
print(f"  Sequences: {len(X_synth):,}")
print(f"  X shape: {X_synth.shape}  (samples, timesteps, features)")
print(f"  y shape: {y_synth.shape}  (samples, features)")

# ==============================================================================
# 4. TEST LSTM MODEL
# ==============================================================================
print("\n" + "=" * 70)
print("  TESTING LSTM (30-MIN AHEAD FORECAST)")
print("=" * 70)

try:
    from tensorflow.keras.models import load_model
    lstm_model = load_model(LSTM_MODEL_PATH)
    print(f"  ✓ LSTM loaded")

    lstm_pred_scaled = lstm_model.predict(X_synth, verbose=0)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
    y_actual = scaler.inverse_transform(y_synth)

    lstm_mae = mean_absolute_error(y_actual, lstm_pred)
    lstm_rmse = np.sqrt(mean_squared_error(y_actual, lstm_pred))
    lstm_mape = np.mean(np.abs((y_actual - lstm_pred) / y_actual)) * 100

    print(f"  MAE:  {lstm_mae:.2f} mg/dL")
    print(f"  RMSE: {lstm_rmse:.2f} mg/dL")
    print(f"  MAPE: {lstm_mape:.2f}%")
    print(f"  Acc:  {100-lstm_mape:.2f}%")

    lstm_results = {'pred': lstm_pred.flatten(), 'actual': y_actual.flatten(),
                    'mae': lstm_mae, 'rmse': lstm_rmse, 'mape': lstm_mape}
except Exception as e:
    print(f"  ✗ LSTM test failed: {e}")
    lstm_results = None

# ==============================================================================
# 5. TEST GRU MODEL
# ==============================================================================
print("\n" + "=" * 70)
print("  TESTING GRU (30-MIN AHEAD FORECAST)")
print("=" * 70)

try:
    gru_model = load_model(GRU_MODEL_PATH)
    print(f"  ✓ GRU loaded")

    gru_pred_scaled = gru_model.predict(X_synth, verbose=0)
    gru_pred = scaler.inverse_transform(gru_pred_scaled)

    gru_mae = mean_absolute_error(y_actual, gru_pred)
    gru_rmse = np.sqrt(mean_squared_error(y_actual, gru_pred))
    gru_mape = np.mean(np.abs((y_actual - gru_pred) / y_actual)) * 100

    print(f"  MAE:  {gru_mae:.2f} mg/dL")
    print(f"  RMSE: {gru_rmse:.2f} mg/dL")
    print(f"  MAPE: {gru_mape:.2f}%")
    print(f"  Acc:  {100-gru_mape:.2f}%")

    gru_results = {'pred': gru_pred.flatten(), 'actual': y_actual.flatten(),
                   'mae': gru_mae, 'rmse': gru_rmse, 'mape': gru_mape}
except Exception as e:
    print(f"  ✗ GRU test failed: {e}")
    gru_results = None

# ==============================================================================
# 6. TEST DENSE MODEL
# ==============================================================================
print("\n" + "=" * 70)
print("  TESTING DENSE (30-MIN AHEAD FORECAST)")
print("=" * 70)

try:
    dense_model = load_model(DENSE_MODEL_PATH)
    print(f"  ✓ Dense loaded")

    dense_pred_scaled = dense_model.predict(X_synth, verbose=0)
    dense_pred = scaler.inverse_transform(dense_pred_scaled)

    dense_mae = mean_absolute_error(y_actual, dense_pred)
    dense_rmse = np.sqrt(mean_squared_error(y_actual, dense_pred))
    dense_mape = np.mean(np.abs((y_actual - dense_pred) / y_actual)) * 100

    print(f"  MAE:  {dense_mae:.2f} mg/dL")
    print(f"  RMSE: {dense_rmse:.2f} mg/dL")
    print(f"  MAPE: {dense_mape:.2f}%")
    print(f"  Acc:  {100-dense_mape:.2f}%")

    dense_results = {'pred': dense_pred.flatten(), 'actual': y_actual.flatten(),
                     'mae': dense_mae, 'rmse': dense_rmse, 'mape': dense_mape}
except Exception as e:
    print(f"  ✗ Dense test failed: {e}")
    dense_results = None

# ==============================================================================
# 7. TEST HMM MODEL (for comparison)
# ==============================================================================
print("\n" + "=" * 70)
print("  TESTING HMM (STATE ESTIMATION - NOT TRUE FORECAST)")
print("=" * 70)

try:
    hmm_model = joblib.load(HMM_MODEL_PATH)
    print(f"  ✓ HMM loaded: {hmm_model.n_components} states")

    # HMM does state estimation on current values, not true forecasting
    # We compare it to the ACTUAL at the same time step (not future)
    state_probs = hmm_model.predict_proba(synthetic_scaled)
    hmm_means = hmm_model.means_.flatten()
    hmm_pred_weighted = (state_probs * hmm_means).sum(axis=1)

    window = 5
    hmm_pred_smooth = np.convolve(hmm_pred_weighted, np.ones(window)/window, mode='same')
    hmm_pred_final = scaler.inverse_transform(hmm_pred_smooth.reshape(-1, 1))

    # Align with actual (same time step, not future)
    actual_aligned = df_ts['bgl'].values[:len(hmm_pred_final)]

    hmm_mae = mean_absolute_error(actual_aligned, hmm_pred_final)
    hmm_rmse = np.sqrt(mean_squared_error(actual_aligned, hmm_pred_final))
    hmm_mape = np.mean(np.abs((actual_aligned - hmm_pred_final.flatten()) / actual_aligned)) * 100

    print(f"  MAE:  {hmm_mae:.2f} mg/dL (current state estimation)")
    print(f"  RMSE: {hmm_rmse:.2f} mg/dL")
    print(f"  MAPE: {hmm_mape:.2f}%")
    print(f"  Acc:  {100-hmm_mape:.2f}%")
    print(f"  ⚠️  NOTE: HMM estimates CURRENT state, not future BGL")

    hmm_results = {'pred': hmm_pred_final.flatten(), 'actual': actual_aligned,
                   'mae': hmm_mae, 'rmse': hmm_rmse, 'mape': hmm_mape}
except Exception as e:
    print(f"  ✗ HMM test failed: {e}")
    hmm_results = None

# ==============================================================================
# 8. SUMMARY COMPARISON
# ==============================================================================
print("\n" + "=" * 70)
print("  SUMMARY: 30-MINUTE AHEAD FORECASTING PERFORMANCE")
print("=" * 70)

results = []
if lstm_results:
    results.append(['LSTM', lstm_results['mae'], lstm_results['rmse'], lstm_results['mape'], 100-lstm_results['mape']])
if gru_results:
    results.append(['GRU', gru_results['mae'], gru_results['rmse'], gru_results['mape'], 100-gru_results['mape']])
if dense_results:
    results.append(['Dense', dense_results['mae'], dense_results['rmse'], dense_results['mape'], 100-dense_results['mape']])
if hmm_results:
    results.append(['HMM*', hmm_results['mae'], hmm_results['rmse'], hmm_results['mape'], 100-hmm_results['mape']])

if results:
    print(f"\n  {'Model':<10} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'Accuracy':>10}")
    print(f"  {'-'*60}")
    for r in results:
        print(f"  {r[0]:<10} {r[1]:>10.2f} {r[2]:>10.2f} {r[3]:>9.2f}% {r[4]:>9.2f}%")

    print("\n  * HMM is state estimation (current), not true 30-min forecast")
    print("\n  INTERPRETATION for 30-min forecasting:")
    print("    MAE < 15  → Excellent for clinical use")
    print("    MAE 15-25 → Good, acceptable for CGM alerts")
    print("    MAE 25-40 → Moderate, useful for trend direction")
    print("    MAE > 40  → Poor, not reliable for decisions")

# ==============================================================================
# 9. PLOTS
# ==============================================================================
print("\n" + "=" * 70)
print("  GENERATING PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('30-Minute Ahead BGL Forecasting on Fresh Synthetic Data', fontsize=14, fontweight='bold')

# Plot 1: LSTM forecast
if lstm_results:
    axes[0,0].plot(lstm_results['actual'][:300], label='Actual', color='black', linewidth=1.5)
    axes[0,0].plot(lstm_results['pred'][:300], label='LSTM 30-min ahead', color='blue', linewidth=1.5, alpha=0.8)
    axes[0,0].set_title(f'LSTM Forecast (MAE={lstm_results["mae"]:.1f})')
    axes[0,0].set_xlabel('Time Step')
    axes[0,0].set_ylabel('BGL (mg/dL)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

# Plot 2: GRU forecast
if gru_results:
    axes[0,1].plot(gru_results['actual'][:300], label='Actual', color='black', linewidth=1.5)
    axes[0,1].plot(gru_results['pred'][:300], label='GRU 30-min ahead', color='green', linewidth=1.5, alpha=0.8)
    axes[0,1].set_title(f'GRU Forecast (MAE={gru_results["mae"]:.1f})')
    axes[0,1].set_xlabel('Time Step')
    axes[0,1].set_ylabel('BGL (mg/dL)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

# Plot 3: All models scatter
if lstm_results and gru_results:
    axes[1,0].scatter(lstm_results['actual'], lstm_results['pred'], alpha=0.3, s=8, color='blue', label='LSTM')
    axes[1,0].scatter(gru_results['actual'], gru_results['pred'], alpha=0.3, s=8, color='green', label='GRU')
    min_v = min(lstm_results['actual'].min(), lstm_results['pred'].min())
    max_v = max(lstm_results['actual'].max(), lstm_results['pred'].max())
    axes[1,0].plot([min_v, max_v], [min_v, max_v], 'r--', label='Perfect')
    axes[1,0].set_title('Actual vs Predicted')
    axes[1,0].set_xlabel('Actual BGL')
    axes[1,0].set_ylabel('Predicted BGL')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

# Plot 4: Residuals for best model
best = None
if lstm_results and gru_results:
    best = lstm_results if lstm_results['mae'] < gru_results['mae'] else gru_results
elif lstm_results:
    best = lstm_results
elif gru_results:
    best = gru_results

if best:
    res = best['actual'] - best['pred']
    axes[1,1].hist(res, bins=40, color='darkorange', edgecolor='black', alpha=0.7)
    axes[1,1].axvline(0, color='black', linestyle='--', label='Zero')
    axes[1,1].axvline(res.mean(), color='red', linestyle='--', label=f'Mean={res.mean():.1f}')
    axes[1,1].set_title(f'Residuals (μ={res.mean():.1f}, σ={res.std():.1f})')
    axes[1,1].set_xlabel('Error (mg/dL)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('forecast_30min_results.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: forecast_30min_results.png")
plt.show()

print("\n" + "=" * 70)
print("  TEST COMPLETE")
print("=" * 70)
