import pandas as pd 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
# Define Preprocessing packages
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
# For defining model
from sklearn.ensemble import RandomForestRegressor
# data path
data_path = "C:/Users/asus/Documents/Learning Machine Learning/melb_data.csv"
data = pd.read_csv(data_path)
# create target object y
y = data.Price
# create feature array and store it in X
features = ['Rooms','Type','Distance','Bedroom2','Bathroom','Car','Landsize','BuildingArea','YearBuilt','CouncilArea','Regionname','Propertycount']
X = data[features]
n = (X.dtypes=='float64')
numerical_cols=list(n[n].index)
c=(X.dtypes=='object')
categorical_cols=list(c[c].index)
# Split data into training and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
# Preprocessing Numerical Data
numerical_tansformer = SimpleImputer(strategy='constant')
# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[('num',numerical_tansformer,numerical_cols),('cat',categorical_transformer,categorical_cols)])
# Define the model
model=RandomForestRegressor(n_estimators=100,random_state=0)
my_pipeline = Pipeline(steps = [('preprocessor',preprocessor),('model',model)])
# Preprocessing training data and fitting the model
my_pipeline.fit(X_train,y_train)
# Preprocessing test data and getting predictions
preds = my_pipeline.predict(X_test)
# Evaluate model using MAE
score=mean_absolute_error(y_test,preds)
print('MAE:',score)