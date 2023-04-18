import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pycaret.regression import setup, compare_models, tune_model, predict_model

# Load data
file_path = "finals/data/SPY_2023-04-14_time_series.csv"
csv_file = pd.read_csv(file_path, parse_dates=["timestamp"])
csv_file["expiration"] = csv_file["expiration"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").timestamp())
csv_file.set_index("timestamp", inplace=True)

# Preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(csv_file.values)
X, y = scaled_data[:, 1:], scaled_data[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Prepare data for PyCaret
train_data = pd.DataFrame(X_train, columns=csv_file.columns[1:])
train_data['spy_price'] = y_train
test_data = pd.DataFrame(X_test, columns=csv_file.columns[1:])
test_data['spy_price'] = y_test

# Set up PyCaret
regression_setup = setup(
    data=train_data,
    target='spy_price',
    train_size=0.8,
    preprocess=False,  # Skip preprocessing as we have already done it
    verbose=False,
    session_id=42
)

# Compare models
best_models = compare_models(n_select=3, sort="R2")

# Hyperparameter tuning for the top 3 models
tuned_models = [tune_model(model, optimize="R2") for model in best_models]

# Make predictions on test_data
predictions = []
for model in tuned_models:
    preds = predict_model(model, data=test_data)
    predictions.append(preds['spy_price'].values)  # Use 'spy_price' instead of 'Label'

# Compare actual and predicted values
actual_vs_predicted = pd.DataFrame({"Actual": test_data["spy_price"].values})
for i, preds in enumerate(predictions):
    actual_vs_predicted[f"Predicted_{i + 1}"] = preds

# Determine next direction (up or down)
next_directions = []
for model_num in range(1, len(predictions) + 1):
    next_direction = ["buy" if p > a else "sell" for a, p in zip(actual_vs_predicted["Actual"], actual_vs_predicted[f"Predicted_{model_num}"])]
    next_directions.append(next_direction)

# Determine if the model got each direction right or wrong
got_direction_right = []
for direction in next_directions:
    right_or_wrong = []
    for i in range(1, len(direction)):
        if direction[i] == direction[i-1]:
            right_or_wrong.append("right")
        else:
            right_or_wrong.append("wrong")
    got_direction_right.append(right_or_wrong)

# Print results
for i, (direction, right_or_wrong) in enumerate(zip(next_directions, got_direction_right), 1):
    print(f"Model {i} direction decisions:")
    print(direction)
    print(f"Model {i} got each direction decision right or wrong:")
    print(right_or_wrong)

# Calculate percentage of right decisions
for i, right_or_wrong in enumerate(got_direction_right, 1):
    right_decisions = right_or_wrong.count("right")
    total_decisions = len(right_or_wrong)
    percentage_right = (right_decisions / total_decisions) * 100
    print(f"Percentage of right decisions for model {i}: {percentage_right:.2f}%")
