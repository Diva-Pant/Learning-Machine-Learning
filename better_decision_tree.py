import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
# Objectives
# create model, split, fit, predict
re_data = pd.read_excel('real_estate_data.xlsx')
# choosing features
re_data.columns = ['No','txn_date','house_age','prox_market','no_stores','lat','long','per_unit_price']
re_features = ['house_age', 'prox_market', 'no_stores', 'lat', 'long']
# print(re_data.head())
# print(re_data.describe())
# split the data
X = re_data[re_features]
y = re_data.per_unit_price
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# Build model
re_model = DecisionTreeRegressor(random_state = 1)
# Fit model with training data
re_model.fit(train_X,train_y)
# make predictions on validation features
re_predict = re_model.predict(val_X)
# print(val_X.head())
# print("The predictions are")
# print(re_model.predict(val_X.head()))
# calculate MAE
val_mae = mean_absolute_error(re_predict, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))