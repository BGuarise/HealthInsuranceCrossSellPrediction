import numpy as np 
import pandas as pd
import joblib

# Function to transform collums with values between few string named classes into integers 
def string_to_bool(data, collum_name, shifft):
	labels = data[collum_name].unique()
	labels = np.flip(labels)
	data[collum_name].replace({labels[k]:k+shifft for k in range(len(labels)) }, inplace = True)
	return data

# Function to transform collums with values disperce in  several string named classes into one-hot-encode representation
def one_hot_state(data, collum_name, all_values):
	one_hot = pd.get_dummies(data[collum_name], prefix= collum_name)
	one_hot = one_hot.T.reindex(all_values).T.fillna(0) #This line fills possible collums os the one-hot-encode that doesn't appear in the data
	data.drop(collum_name, axis = 1)
	data = data.join(one_hot)
	return data

# Function to perform min-max normalization
def min_max_normalization(data, collum_name):
	data[collum_name] = (data[collum_name] - data[collum_name].min()) / (data[collum_name].max() - data[collum_name].min())
	return data

def data_transformation(data):
	
	
	data = string_to_bool(data, 'Gender', 0)
	data = string_to_bool(data, 'Vehicle_Damage', 0)
	data = string_to_bool(data, 'Vehicle_Age', 1)

	data = one_hot_state(data, 'Region_Code', ['Region_Code' + str(i) for i in range(53)])
	data = one_hot_state(data, 'Policy_Sales_Channel', ['Policy_Sales_Channel' + str(i) for i in range(1,164,1)])

	data = min_max_normalization(data, 'Annual_Premium')
	data = min_max_normalization(data, 'Age')
	data = min_max_normalization(data, 'Vintage')

	data.drop('id', axis = 1, inplace = True)
	return data
