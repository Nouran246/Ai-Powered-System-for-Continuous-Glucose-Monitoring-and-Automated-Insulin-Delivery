"""
================================================================================
SAVE MODELS & SCALER FOR TESTING
================================================================================
Add this to the END of your training notebook to save everything needed
for the test scripts.

This ensures the test scripts use the EXACT same scaler and models.
================================================================================
"""

import joblib
from tensorflow.keras.models import save_model

# 1. SAVE SCALER (critical!)
joblib.dump(scaler, 'scaler.pkl')
print("✓ scaler.pkl saved")

# 2. SAVE KERAS MODELS
lstm_model.save('lstm_model.keras')
print("✓ lstm_model.keras saved")

gru_model.save('gru_model.keras')
print("✓ gru_model.keras saved")

dense_model.save('dense_model.keras')
print("✓ dense_model.keras saved")

# 3. SAVE HMM
joblib.dump(hmm_model, 'hmm_model.pkl')
print("✓ hmm_model.pkl saved")

# 4. SAVE TRAINING METRICS (optional, for comparison)
import json
training_info = {
    "window_size": WINDOW_SIZE,
    "horizon": HORIZON,
    "data_frequency": "30min",
    "scaler_min": float(scaler.data_min_[0]),
    "scaler_max": float(scaler.data_max_[0]),
    "train_size": len(train_data),
    "test_size": len(test_data)
}
with open('training_info.json', 'w') as f:
    json.dump(training_info, f, indent=2)
print("✓ training_info.json saved")

print("\nAll files ready for testing with test_30min_forecast.py!")
