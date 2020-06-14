import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
re_data = pd.read_excel('real_estate_data.xlsx')
# choosing features
re_data.columns = ['No','txn_date','house_age','prox_market','no_stores','lat','long','per_unit_price']
# create y 
y = re_data.per_unit_price
# create X
features = ['house_age', 'prox_market', 'no_stores', 'lat', 'long']
X = re_data[features]
# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 1)
# Specify model
re_model = DecisionTreeRegressor(random_state=1)
# Fit model
re_model.fit(X_train,y_train)
# make predictions and calculate MAE
re_predictions = re_model.predict(X_test)
re_mae = mean_absolute_error(re_predictions, y_test)
print("Validation MAE:{:,.0f}".format(re_mae))
# mae function to compare MAE scores from different values for max_leaf_nodes
# solving underfitting and overfitting using Decision Tree
def get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    preds_val = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds_val)
    return(mae)
# compare different tree sizes
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size, X_train, X_test, y_train, y_test) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)
# fit predict and score model with best tree size
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
final_model.fit(X,y)
final_model.predict(X_test)
mean_absolute_error(final_model.predict(X_test),y_test)
print("Validation MAE:{:,.0f}".format(mean_absolute_error(final_model.predict(X_test),y_test)))
# The result are better, but much efficient altrnative would be Random Forest