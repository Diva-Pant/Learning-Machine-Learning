import pandas as pd
import numpy as np
voting_path = "C:/Users/asus/Documents/Learning Machine Learning/house-votes-84.data"
# Adding labels to the dataframe according to the data attributes
labels = ['Party', 'h_infants', 'water_cost', 'adopt_budget', 'physician', 'salvador', 'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels', 'education', 'superfund', 'crime', 'duty_free_exp', 'export_SA']
missing_values = ["?"]
voting_records = pd.read_csv(voting_path, names = labels, na_values = missing_values)
# perform EDA head() info() describe()
# print(voting_records)
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
# Assumption members of same party have generally same vote
demo = voting_records.groupby(['Party']).get_group('democrat')
repub = voting_records.groupby(['Party']).get_group('republican')
for col in demo.columns:
	demo[col].replace('?',np.nan,inplace=True)
	demo[col].fillna(demo[col].mode()[0], inplace=True)
for col in repub.columns:
	repub[col].replace('?',np.nan,inplace=True)
	repub[col].fillna(repub[col].mode()[0], inplace=True)
# dataset without missing values concat
dataset = pd.concat([demo,repub])
print(dataset)