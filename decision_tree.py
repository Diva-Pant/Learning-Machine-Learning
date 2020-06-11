import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
# Objectives
# create model, fit, predict
# use decision tree regressor
re_data = pd.read_excel('real_estate_data.xlsx')
# choosing features
re_data.columns = ['No','txn_date','house_age','prox_market','no_stores','lat','long','per_unit_price']
re_features = ['house_age', 'prox_market', 'no_stores', 'lat', 'long']
# print(re_data.head())
# print(re_data.describe())
# Build model
re_model = DecisionTreeRegressor(random_state = 1)
X = re_data[re_features]
y = re_data.per_unit_price
# Fit model
re_model.fit(X,y)
# make predictions
# in-sample prediction, not so effective with new training examples
print(X.head())
print("The predictions are")
print(re_model.predict(X.head()))
