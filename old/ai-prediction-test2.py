import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the CSV data into a pandas dataframe
df = pd.read_csv("finals/data/SPY_2023-04-14_time_series.csv")

# Select the relevant columns
X = df[['vix', 'pcr', 'put_notional', 'call_notional']]
y = df['spy_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the linear regression model on the training data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Use the trained model to make predictions on the test data
y_pred = regressor.predict(X_test)

# Evaluate the model's performance
error = np.mean(np.abs(y_pred - y_test))
print("Mean absolute error: ", error)

# Use the trained model to predict the next price of SPY
next_price = regressor.predict([[17.8, 0.5, 3808039, 6746493]])
print("Next price of SPY: ", next_price)
