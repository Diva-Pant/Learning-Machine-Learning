import pandas as pd 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
re_data = pd.read_excel('real_estate_data.xlsx')
re_data.columns = ['No','txn_date','house_age','prox_market','no_stores','lat','long','per_unit_price']
re_features = ['house_age', 'prox_market', 'no_stores', 'lat', 'long']
# target object y
y = re_data.per_unit_price
# create X
X = re_data[re_features]
# split into test and training data
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)
# specify model for regression tree
# -----------------------------------------------------------------------------------------
re_model = DecisionTreeRegressor(random_state=1)
# fit model
re_model.fit(X_train,y_train)
# make predictions
re_predictions = re_model.predict(X_test)
# calculate MAE
re_mae = mean_absolute_error(re_predictions,y_test)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(re_mae))
# using best value for max_leaf_nodes
re_model = DecisionTreeRegressor(max_leaf_nodes=100,random_state=1)
re_model.fit(X_train,y_train)
re_predictions = re_model.predict(X_test)
re_mae = mean_absolute_error(re_predictions,y_test)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(re_mae))
# -----------------------------------------------------------------------------------------
# specify model for random forest
rf_model = RandomForestRegressor()
# fit the model
rf_model.fit(X_train,y_train)
# make predictions
rf_predictions = rf_model.predict(X_test)
rf_mae = mean_absolute_error(rf_predictions, y_test)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(rf_mae))
# Random forst gives better results