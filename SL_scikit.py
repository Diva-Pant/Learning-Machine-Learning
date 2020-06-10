import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import math
voting_path = "C:/Users/asus/Documents/Learning Machine Learning/house-votes-84.data"
# Adding labels to the dataframe according to the data attributes
labels = ['Party', 'h_infants', 'water_cost', 'adopt_budget', 'physician', 'salvador', 'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels', 'education', 'superfund', 'crime', 'duty_free_exp', 'export_SA']
missing_values = ["?"]
voting_records = pd.read_csv(voting_path, names = labels, na_values = missing_values)
# perform EDA head() info() describe()
# print(voting_records.head())
# print(voting_records.info())
# print(voting_records.describe())
# To print the column names
# for col in voting_records.columns:
	# print(col)
# Data preprocessing
# Dealing with the missing values in pandas
# Categorical encoding, using find and replace method/onehot encoding
for i in labels:
	cleanup = {i : {"n" : 0, "y" : 1}}
	voting_records.replace(cleanup, inplace = True)
# print(voting_records.head().round())
# print(voting_records.info())
# print(voting_records.describe().round())
# replacing missing values, removing rows is not an option as many rows has missing value
# Imputation method - Using k-NN algorithm
# Assumption members of same party have generally same vote
# Taking odd k if n is evento avoid binary classes ties, vice-versa
demo = voting_records.groupby(['Party']).get_group('democrat')
repub = voting_records.groupby(['Party']).get_group('republican')
# print(demo.describe().round())
# print(repub.describe().round())
# print(demo.info())
# print(demo.head())
# print(repub.head())
# Replacing ? with NAN so that the KNN imputer recognoze it as missing value using numpy for it
# Create KNN imputer
# choosing right k, sqrt(n)
n1=math.floor(math.sqrt(demo['Party'].count()))
# print(n1)
imputer1 = KNNImputer(n_neighbors=n1)
filling_vald = imputer1.fit_transform(demo.drop('Party', axis = 1))
# print(filling_vald)
n2=math.floor(math.sqrt(repub['Party'].count()))
# print(n2)
imputer2 = KNNImputer(n_neighbors=n2)
filling_valr = imputer2.fit_transform(repub.drop('Party', axis = 1))
