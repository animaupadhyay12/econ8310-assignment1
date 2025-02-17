# %%
import pandas as pd

train = pd.read_csv('assignment_data_train.csv')
test = pd.read_csv('assignment_data_test.csv')

# %%
train.head()

# %%
import statsmodels.api as sm


x = train.drop(['trips', 'Timestamp'], axis = 1)
y = train[['trips']]

# %%
model = sm.OLS(endog = y, exog = x)
modelFit = model.fit()

# %%
x_test = test.drop(['Timestamp'], axis = 1)

pred = modelFit.predict(x_test)


