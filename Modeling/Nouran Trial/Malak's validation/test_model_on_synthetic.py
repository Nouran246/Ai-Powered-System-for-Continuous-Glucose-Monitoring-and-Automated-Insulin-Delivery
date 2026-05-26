"""
================================================================================
SYNTHETIC BGL MODEL TEST SCRIPT
================================================================================
This script loads your saved HMM model (hmm_model.pkl) and tests it on 
fresh synthetic blood glucose data that the model has NEVER seen before.

The synthetic data was generated to match the statistical properties of your
training data (mean, std, autocorrelation, hourly patterns) but uses entirely
new timestamps (Jan-Feb 2026) and a different random seed.

REQUIREMENTS:
    pip install pandas numpy scikit-learn joblib matplotlib

USAGE:
    python test_model_on_synthetic.py

OUTPUT:
    - Console metrics (MAE, RMSE, MAPE, Accuracy)
    - Prediction vs Actual plot
    - Residual analysis plot
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
MODEL_PATH = 'hmm_model.pkl'                    # Your saved HMM model
SYNTHETIC_DATA_PATH = 'synthetic_bgl_test_data.csv'  # Fresh synthetic data
WINDOW_SIZE = 24                                # 2 hours of history (from notebook)
HORIZON = 6                                     # 30-minute prediction horizon
MEDICAL_MIN = 40
MEDICAL_MAX = 400

# ==============================================================================
# 1. LOAD SYNTHETIC DATA
# ==============================================================================
print("=" * 70)
print("  LOADING FRESH SYNTHETIC TEST DATA")
print("=" * 70)

df = pd.read_csv(SYNTHETIC_DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"\nSynthetic data loaded:")
print(f"  Records: {len(df):,}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
print(f"  Duration: {(df['date'].max() - df['date'].min()).days} days")
print(f"  BGL mean: {df['bgl'].mean():.2f} mg/dL")
print(f"  BGL std:  {df['bgl'].std():.2f} mg/dL")

# ==============================================================================
# 2. PREPROCESS (same pipeline as training)
# ==============================================================================
print("\n" + "=" * 70)
print("  PREPROCESSING (matching training pipeline)")
print("=" * 70)

# Set index
df_ts = df.set_index('date')[['bgl']].copy()

# Clip to medical bounds (same as training)
bgl_before = df_ts['bgl'].copy()
df_ts['bgl'] = df_ts['bgl'].clip(lower=MEDICAL_MIN, upper=MEDICAL_MAX)
clipped = (bgl_before != df_ts['bgl']).sum()
print(f"  Clipped {clipped} values to medical bounds [{MEDICAL_MIN}, {MEDICAL_MAX}]")

# ==============================================================================
# 3. SCALE THE DATA
# ==============================================================================
# IMPORTANT: In the original notebook, the scaler was fit on TRAINING data only.
# For a true out-of-sample test, you should use the SAME scaler that was fit 
# during training. If you saved it, load it here. Otherwise, we fit on the 
# synthetic data (which is NOT ideal but works for demonstration).
#
# RECOMMENDATION: Save your scaler during training with:
#   joblib.dump(scaler, 'scaler.pkl')
# Then load it here with:
#   scaler = joblib.load('scaler.pkl')

print("\n  Fitting MinMaxScaler on synthetic data...")
print("  ⚠️  NOTE: For best results, use the SAME scaler from training!")
scaler = MinMaxScaler()
synthetic_scaled = scaler.fit_transform(df_ts[['bgl']])

print(f"  Scaler min: {scaler.data_min_[0]:.2f}")
print(f"  Scaler max: {scaler.data_max_[0]:.2f}")

# ==============================================================================
# 4. LOAD HMM MODEL
# ==============================================================================
print("\n" + "=" * 70)
print("  LOADING HMM MODEL")
print("=" * 70)

try:
    hmm_model = joblib.load(MODEL_PATH)
    print(f"  ✓ Model loaded from {MODEL_PATH}")
    print(f"  States: {hmm_model.n_components}")
    print(f"  Covariance type: {hmm_model.covariance_type}")
except FileNotFoundError:
    print(f"  ✗ ERROR: Model file not found at {MODEL_PATH}")
    print(f"    Please ensure hmm_model.pkl is in the same directory.")
    exit(1)
except Exception as e:
    print(f"  ✗ ERROR loading model: {e}")
    exit(1)

# ==============================================================================
# 5. CREATE SEQUENCES (same as training)
# ==============================================================================
print("\n" + "=" * 70)
print("  CREATING PREDICTION SEQUENCES")
print("=" * 70)

def create_sequences(data, window_size, horizon=1):
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size+horizon-1])
    return np.array(X), np.array(y)

X_synth, y_synth = create_sequences(synthetic_scaled, WINDOW_SIZE, horizon=HORIZON)
print(f"  Sequences created: {len(X_synth):,}")
print(f"  X shape: {X_synth.shape}")
print(f"  y shape: {y_synth.shape}")

# ==============================================================================
# 6. RUN PREDICTIONS
# ==============================================================================
print("\n" + "=" * 70)
print("  RUNNING HMM PREDICTIONS ON SYNTHETIC DATA")
print("=" * 70)

# HMM prediction pipeline (same as notebook)
state_probs = hmm_model.predict_proba(synthetic_scaled)
hmm_means = hmm_model.means_.flatten()

# Weighted prediction
hmm_pred_weighted = (state_probs * hmm_means).sum(axis=1)

# Smoothing (window=5, same as notebook)
window = 5
hmm_pred_smooth = np.convolve(hmm_pred_weighted, np.ones(window)/window, mode='same')

# Inverse transform to original scale
hmm_pred_final = scaler.inverse_transform(hmm_pred_smooth.reshape(-1, 1))

# Align with actual values (same alignment logic as notebook)
actual_bgl = df_ts['bgl'].values[:len(hmm_pred_final)]

print(f"  Predictions generated: {len(hmm_pred_final):,}")

# ==============================================================================
# 7. EVALUATE
# ==============================================================================
print("\n" + "=" * 70)
print("  EVALUATION RESULTS")
print("=" * 70)

mae = mean_absolute_error(actual_bgl, hmm_pred_final)
rmse = np.sqrt(mean_squared_error(actual_bgl, hmm_pred_final))
mape = np.mean(np.abs((actual_bgl - hmm_pred_final.flatten()) / actual_bgl)) * 100
accuracy = 100 - mape

print(f"\n  📊 HMM ON FRESH SYNTHETIC DATA:")
print(f"     MAE      : {mae:.2f} mg/dL")
print(f"     RMSE     : {rmse:.2f} mg/dL")
print(f"     MAPE     : {mape:.2f}%")
print(f"     Accuracy : {accuracy:.2f}%")

# Sanity check
print(f"\n  🔍 Sanity Check:")
print(f"     Actual min/max : {actual_bgl.min():.1f} / {actual_bgl.max():.1f}")
print(f"     Pred  min/max  : {hmm_pred_final.min():.1f} / {hmm_pred_final.max():.1f}")
print(f"     Actual mean    : {actual_bgl.mean():.1f}")
print(f"     Pred  mean     : {hmm_pred_final.mean():.1f}")

# ==============================================================================
# 8. VISUALIZATIONS
# ==============================================================================
print("\n" + "=" * 70)
print("  GENERATING PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('HMM Model Test on Fresh Synthetic BGL Data', fontsize=14, fontweight='bold')

# Plot 1: Full prediction vs actual
axes[0,0].plot(actual_bgl, label='Actual BGL', color='black', alpha=0.8, linewidth=1)
axes[0,0].plot(hmm_pred_final, label='HMM Prediction', color='red', alpha=0.7, linewidth=1)
axes[0,0].set_title('Prediction vs Actual (Full Series)')
axes[0,0].set_xlabel('Time Step')
axes[0,0].set_ylabel('BGL (mg/dL)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Zoomed first 500 points
zoom = 500
axes[0,1].plot(actual_bgl[:zoom], label='Actual', color='black', linewidth=1.5)
axes[0,1].plot(hmm_pred_final[:zoom], label='HMM', color='red', linewidth=1.5, alpha=0.8)
axes[0,1].set_title(f'Zoomed View (First {zoom} Points)')
axes[0,1].set_xlabel('Time Step')
axes[0,1].set_ylabel('BGL (mg/dL)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Scatter (actual vs predicted)
axes[1,0].scatter(actual_bgl, hmm_pred_final, alpha=0.3, s=10, color='purple')
min_val = min(actual_bgl.min(), hmm_pred_final.min())
max_val = max(actual_bgl.max(), hmm_pred_final.max())
axes[1,0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
axes[1,0].set_title('Actual vs Predicted Scatter')
axes[1,0].set_xlabel('Actual BGL (mg/dL)')
axes[1,0].set_ylabel('Predicted BGL (mg/dL)')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Residuals
residuals = actual_bgl - hmm_pred_final.flatten()
axes[1,1].hist(residuals, bins=50, color='darkorange', edgecolor='black', alpha=0.7)
axes[1,1].axvline(0, color='black', linestyle='--', linewidth=1.5, label='Zero')
axes[1,1].axvline(residuals.mean(), color='red', linestyle='--', linewidth=1.5, 
                   label=f'Mean={residuals.mean():.1f}')
axes[1,1].set_title(f'Residual Distribution (μ={residuals.mean():.2f}, σ={residuals.std():.2f})')
axes[1,1].set_xlabel('Residual (mg/dL)')
axes[1,1].set_ylabel('Frequency')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('synthetic_test_results.png', dpi=150, bbox_inches='tight')
print("  ✓ Plot saved to: synthetic_test_results.png")
plt.show()

# ==============================================================================
# 9. COMPARISON WITH TRAINING PERFORMANCE (if available)
# ==============================================================================
print("\n" + "=" * 70)
print("  INTERPRETATION GUIDE")
print("=" * 70)
print("""
How to interpret these results:

1. MAE < 20 mg/dL  → Excellent generalization
2. MAE 20-35 mg/dL → Good generalization  
3. MAE 35-50 mg/dL → Moderate, may need retraining
4. MAE > 50 mg/dL  → Poor, model likely overfit

Compare these metrics to your original test-set metrics:
   - If similar → Model generalizes well ✓
   - If much worse → Model overfit to training data ✗
   - If much better → Synthetic data may be too easy (check distribution)

NOTE: The synthetic data preserves:
   - Hourly BGL patterns
   - Autocorrelation structure  
   - Medical bounds (40-400)
   - Realistic meal/correction spikes

But it does NOT include:
   - The exact same patient behaviors
   - The same date range (uses Jan-Feb 2026)
   - The same random seed
""")

print("\n" + "=" * 70)
print("  TEST COMPLETE")
print("=" * 70)
