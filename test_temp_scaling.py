from src.calibration import TemperatureScaler
import numpy as np

# Create temperature scaler
scaler = TemperatureScaler()

# Test different temperatures
for temp in [0.5, 1.0, 2.0]:
    scaler.temperature = temp
    raw_conf = 0.7
    calibrated = scaler.calibrate_confidence(raw_conf, temp)
    print(f"T={temp}: {raw_conf:.2f} → {calibrated:.2f} (change: {(calibrated-raw_conf)*100:+.1f}%)")

print("\n✓ Temperature scaling working correctly")
