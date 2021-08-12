import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection._split import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve

def string_to_bool(data, collum_name, shifft):
	labels = data[collum_name].unique()
	labels = np.flip(labels)
	data[collum_name].replace({labels[k]:k+shifft for k in range(len(labels)) }, inplace = True)
	return data

def one_hot_state(data, collum_name):
	one_hot = pd.get_dummies(data[collum_name], prefix= collum_name)
	data.drop(collum_name, axis = 1)
	data = data.join(one_hot)
	return data

def min_max_normalization(data, collum_name):
	data[collum_name] = (data[collum_name] - data[collum_name].min()) / (data[collum_name].max() - df[collum_name].min())
	return data

def data_transformation(data):
	
	data = string_to_bool(data, 'Gender', 0)
	data = string_to_bool(data, 'Vehicle_Damage', 0)
	data = string_to_bool(data, 'Vehicle_Age', 1)
	data = one_hot_state(data, 'Region_Code')
	data = one_hot_state(data, 'Policy_Sales_Channel')
	data = min_max_normalization(data, 'Annual_Premium')
	data = min_max_normalization(data, 'Age')
	data = min_max_normalization(data, 'Vintage')
	data.drop('id', axis = 1, inplace = True)
	return data


df = pd.read_csv("Insurance_data/train.csv")
df_test = pd.read_csv("Insurance_data/test.csv")

# Data Transformation

df = data_transformation(df)
df_test = data_transformation(df_test)


Y = df['Response']
X = df.drop('Response', axis = 1)

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.3)

n_estimators = 10
max_depth = 3

Logistic_Regression = LogisticRegression(solver="liblinear",class_weight={1:2}, random_state=10)
Logistic_Regression.fit(X_train, Y_train)

random_forest = RandomForestClassifier(
    n_estimators=n_estimators, max_depth=max_depth, random_state=10)
random_forest.fit(X_train, Y_train)

gradient_boosting = GradientBoostingClassifier(
    n_estimators=n_estimators, max_depth=max_depth, random_state=10)
_ = gradient_boosting.fit(X_train, Y_train)

fig, ax = plt.subplots()

models = [
    ("RF", random_forest),
    ("LR", Logistic_Regression),
    ("GBDT", gradient_boosting),
]

model_displays = {}
for name, model in models:
    model_displays[name] = plot_roc_curve(
        model, X_test, Y_test, ax=ax, name=name)
_ = ax.set_title('ROC curve')

plt.show() 