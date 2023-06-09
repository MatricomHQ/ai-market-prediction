import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

# rather than printing, we might use logging instead
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='runtime.log'
)

# Load data
file_path = "finals/data/SPY_2023-04-14_time_series.csv"
csv_file = pd.read_csv(
    file_path,
    parse_dates=["timestamp"]
)
csv_file["expiration"] = csv_file["expiration"].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d").timestamp()
)
csv_file.set_index("timestamp", inplace=True)

# Preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(csv_file.values)
X, y = scaled_data[:, 1:], scaled_data[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Reshape data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

logging.info(f"X_test shape: {X_test.shape}")
logging.info(f"y_test shape: {y_test.shape}")

# Create and compile LSTM model with Adagrad optimizer
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer=tf.keras.optimizers.Adagrad(
        learning_rate=0.002
    ),
    loss='binary_crossentropy'
)

# Train the model
model.fit(
    X_train,
    y_train,
    epochs=1,
    batch_size=32,
    validation_data=(
        X_test,
        y_test
    )
)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
logging.info(f"Loss: {loss}")

# Make predictions
predictions = model.predict(X_test)

logging.info(f"predictions shape: {predictions.shape}")

# Inverse transform the predictions
predictions = scaler.inverse_transform(
    np.hstack(
        [
            predictions,
            X_test.reshape(
                X_test.shape[0], -1
            )
        ]
    )
)[:, 0]

# Compare actual and predicted values
_, _, _, _, test_indices, _ = train_test_split(X, y, csv_file.index, test_size=0.2, random_state=42)

actual_vs_predicted = pd.DataFrame({"Actual": csv_file.loc[test_indices, "spy_price"].values[:predictions.shape[0]], "Predicted": predictions})

# Determine next direction (up or down)
next_direction = ["buy" if p > a else "sell" for a, p in zip(actual_vs_predicted["Actual"], actual_vs_predicted["Predicted"])]

# Determine if the model got each direction right or wrong

""" 
List comprehension is faster than a for loop in this case, but it's not as readable.
in general, as the length of the list grows larger we should use a generator or a 
comprehension with a conditional expression instead of a for loop for performance
"""
# got_direction_right = [
#     "right" if next_direction[i] == next_direction[i-1]
#     else "wrong" for i in range(1, len(next_direction))
# ]

"""
instead, to keep this readable, we implement a generator function to get the same result
"""


def get_direction_right():
    """
    we might avoid using else here, these tend to be dirty solutions -
    guido himself says: "if you need to use else, you're doing it wrong"
    """
    for i in range(1, len(next_direction)):
        if next_direction[i] == next_direction[i - 1]:
            yield "right"
        # this is a better solution because the case for wrong is handled explicitly
        if next_direction[i] != next_direction[i - 1]:
            yield "wrong"


got_direction_right = list(get_direction_right())

# Print results
logging.info("All direction decisions:")
logging.info(next_direction)
# print("\nGot each direction decision right or wrong:")
# print(got_direction_right)

# Calculate percentage of right decisions
right_decisions = got_direction_right.count("right")
total_decisions = len(got_direction_right)
percentage_right = (right_decisions / total_decisions) * 100
logging.info(f"Percentage of right decisions: {percentage_right:.2f}%")
