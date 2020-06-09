import pandas as pd
# save filepath to variable
melbourne_path = 'C:/Users/asus/Documents/Learning Machine Learning/melb_data.csv'
# read and store data in dataframe
melbourne_data = pd.read_csv(melbourne_path)
# Summary of data using describe()
# describe () head() return only the dataframe, to view it in cmd use print()
print(melbourne_data.describe())
print(melbourne_data.head())
# average price of houses in melbourne using mean()
avg_house_price = melbourne_data['Price'].mean()
print(avg_house_price)
# rounded average price to one digit using round()
print(avg_house_price.round(1))
# average number of bedrooms in the house
print((melbourne_data['Bedroom2'].mean()).round())
# how old is the newest house
# most recent & oldest date of dataset
recent_date = melbourne_data['Date'].max()
# oldest_date = melbourne_data['Date'].min()
# current date
# current_date = pd.datetime.now().strftime("%d/%m/%Y")
current_date = pd.to_datetime('now').strftime("%d/%m/%Y")
print(current_date)
print(recent_date)
# convert the date from str type to datetime type
r_d = pd.to_datetime(recent_date)
c_d = pd.to_datetime(current_date)
house_age = (c_d - r_d).days
print(house_age)