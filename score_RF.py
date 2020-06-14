import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# Read data
re_data = pd.read_excel('real_estate_data.xlsx', index_col='No')
re_data.columns = ['txn_date','house_age','prox_market','no_stores','lat','long','per_unit_price']
re_features = ['house_age', 'prox_market', 'no_stores', 'lat', 'long']
y = re_data.per_unit_price
X = re_data[re_features]
# split test set from training set
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)
X_train.head()
# evaluate several models
# -------------------------------------------------------------------------------------------
model_1 = RandomForestRegressor(n_estimators=50,random_state=0)
model_2 = RandomForestRegressor(n_estimators=100,random_state=0)
model_3 = RandomForestRegressor(n_estimators=100,criterion='mae',random_state=0)
model_4 = RandomForestRegressor(n_estimators=200,min_samples_split=20,random_state=0)
model_5 = RandomForestRegressor(n_estimators=100,max_depth=7,random_state=0)
models = [model_1,model_2,model_3,model_4,model_5]
# functions for comparing different models
def score_model(model,X_t=X_train,X_v=X_test,y_t=y_train,y_v=y_test):
	model.fit(X_t,y_t)
	prediction = model.predict(X_v)
	return mean_absolute_error(y_v,prediction)
for i in range(0,len(models)):
	mae = score_model(models[i])
	print("Model %d MAE: %d" % (i+1, mae))
# appears that best model is model 4
# -------------------------------------------------------------------------------------------
# generate test predictions
rf_model = RandomForestRegressor()
# fit model
rf_model.fit(X,y)
# find predictions
rf_predict = rf_model.predict(X_test)
# save predictions in format used for submitting the files in csv:
rf_output = pd.DataFrame({'No':X_test.index,'per_unit_price':rf_predict})
rf_output.to_csv('result.csv', index=False)
print(rf_output.head())
