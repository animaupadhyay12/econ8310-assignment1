# -*- coding: utf-8 -*-
"""assignment1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lG5h-EsvS6gDJD1z0u_NMyFPCE7XGmzv
"""

from google.colab import drive
drive.mount('/content/drive')

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX


# Load the training and test data
train = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/assignment_data_train.csv")
test = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/assignment_data_test.csv")

# Remove timestamp column if present
if 'Timestamp' in train.columns:
    train = train.drop(columns=['Timestamp'])
if 'Timestamp' in test.columns:
    test = test.drop(columns=['Timestamp'])

# Ensure all columns are numeric
train = train.apply(pd.to_numeric, errors='coerce')
test = test.apply(pd.to_numeric, errors='coerce')

# Identify and drop constant columns (columns with only one unique value)
constant_cols = [col for col in train.columns if train[col].nunique() == 1]
train = train.drop(columns=constant_cols)
test = test.drop(columns=constant_cols, errors='ignore')

# Drop missing values
train = train.dropna()
test = test.dropna()

# Differencing to ensure stationarity
train_diff = train.diff().dropna()

# Fit the VARMA model (using VARMAX in statsmodels)
model = VARMAX(train_diff, order=(1, 1), enforce_stationarity=False, enforce_invertibility=False)

# Fit the model
modelFit = model.fit(disp=False)

# Forecast for 744 hours
pred_diff = modelFit.forecast(steps=744)

# Convert differenced predictions back to original scale
last_values = train.iloc[-1]  # Get last known values before differencing
pred = pred_diff.cumsum() + last_values['trips']

# Convert predictions into DataFrame
pred_df = pd.DataFrame(pred, columns=['trips'])

# Save predictions to Google Drive
output_path = "/content/drive/My Drive/predictions_varma.csv"
pred_df.to_csv(output_path, index=False)

print(f"VARMA model executed successfully. Predictions saved as '{output_path}'.")