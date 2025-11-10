import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Load CSVs ---
eye = pd.read_csv("eye_data.csv")
eeg = pd.read_csv("eeg_data.csv")

# --- Sort by timestamp ---
eye = eye.sort_values("timestamp")
eeg = eeg.sort_values("timestamp")

# --- Merge by nearest timestamp ---
merged = pd.merge_asof(eye, eeg, on="timestamp", direction="nearest")

# Optional: drop rows with missing values
merged = merged.dropna()

# --- Add target variable (placeholder) ---

# Replace this with actual stress/focus labels later
merged['stress_level'] = [0.5] * len(merged)

# --- Define features and target ---
# You may need to adjust column names depending on your CSVs
# For example, pupil_diameter from eye CSV becomes pupil_diameter_x after merge
feature_cols = [
    'left_x', 'left_y', 'right_x', 'right_y', 'pupil_diameter_x',  # eye features
    'EEG_ch1', 'EEG_ch2', 'EEG_ch3', 'EEG_ch4', 'pupil_diameter_y'  # EEG + pupil
]

X = merged[feature_cols]
y = merged['stress_level']

# --- Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train XGBoost regressor ---
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4
)

model.fit(X_train, y_train)

# --- Predict and evaluate ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Optional: save trained model
model.save_model("xgb_stress_model.json")
print("Trained XGBoost model saved to xgb_stress_model.json")
