# -*- coding: utf-8 -*-
"""assignment1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lG5h-EsvS6gDJD1z0u_NMyFPCE7XGmzv
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the training and test data
train = pd.read_csv('/assignment_data_train.csv')
test = pd.read_csv('/assignment_data_test.csv')

# Remove timestamp column (not needed for forecasting)
if 'Timestamp' in train.columns:
    train = train.drop(columns=['Timestamp'])
if 'Timestamp' in test.columns:
    test = test.drop(columns=['Timestamp'])

# Ensure all columns are numeric
train = train.apply(pd.to_numeric, errors='coerce')
test = test.apply(pd.to_numeric, errors='coerce')

# Drop any missing values
train = train.dropna()
test = test.dropna()

# Target variable (trips)
y_train = train['trips']

# Fit the Exponential Smoothing model
model = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=24)
modelFit = model.fit()

# Forecast for 744 hours (one month of hourly data)
pred = modelFit.forecast(steps=744)

# Convert predictions to DataFrame
pred_df = pd.DataFrame(pred, columns=['trips'])

# Print model summary
print(modelFit.summary())

