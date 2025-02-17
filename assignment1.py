# -*- coding: utf-8 -*-
"""assignment1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lG5h-EsvS6gDJD1z0u_NMyFPCE7XGmzv
"""

import pandas as pd

train = pd.read_csv('/assignment_data_train.csv')
test = pd.read_csv('/assignment_data_test.csv')

train.head()



import statsmodels.api as sm


x = train.drop(['trips', 'Timestamp'], axis = 1)
y = train[['trips']]
model = sm.OLS(endog = y, exog = x)
modelFit = model.fit()

# model = sm.OLS(endog = y, exog = x)
# modelFit = model.fit()

x_test = test.drop(['Timestamp'], axis = 1)

pred = modelFit.predict(x_test)

print(modelFit.summary())

