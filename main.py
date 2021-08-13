# General imports
import numpy as np 
import pandas as pd
import joblib
import os
import utils as ut

# Model imports
from sklearn.model_selection._split import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Metrics/ploting Imports
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix, roc_auc_score, plot_roc_curve

"""
Data Extraction

Collums:
id; Gender; Age; Driving_License; Region_Code; Previously_Insured; Vehicle_Age; Vehicle_Damage;	Annual_Premium; PolicySalesChannel;	Vintage; Response


Dados obtidos na url=https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction
"""
df = pd.read_csv("Insurance_data/train.csv")
df_test = pd.read_csv("Insurance_data/test.csv")

"""
# Data Transformation
"""
df = ut.data_transformation(df)
df_test = ut.data_transformation(df_test)

Y = df['Response']
X = df.drop('Response', axis = 1)

"""
Separation in train and test subsets
"""
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.3)

"""
Model chosen to be used in the API - Logistic Regression

logistic Regression as chosen because we can deal with the
weight of the labels directly and use as trick to help with 
severe unbalanced proportion of 0 and 1 Responses 

"""
Logistic_Regression = LogisticRegression(solver="liblinear",class_weight={1:2}, random_state=10)

"""
Other models tested
"""
"""
n_estimators = 10
max_depth = 503
#knn_classifier = KNeighborsClassifier(3)
#decision_tree = DecisionTreeClassifier(random_state=0, max_depth=max_depth)
#random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=10)
#gradient_boosting = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=10)
#adaboostclassifier = AdaBoostClassifier(n_estimators=n_estimators, random_state=10)
"""

fig, ax = plt.subplots()

models = [
    #("RF", random_forest),
    ("LR", Logistic_Regression),
    #("GBDT", gradient_boosting),
    #("ABC", adaboostclassifier),
    #("DT", decision_tree)
    #("KNN", knn_classifier)
]

# Kfold for Cross-Validation
kf = KFold(n_splits=10, random_state=1, shuffle=True)
scores = []

model_displays = {}
for name, model in models:
	
	#train models
	print(name)
	model.fit(X_train, Y_train)

	#Save than in models/ directory
	joblib.dump(model, 'models/' + name + '.pkl')

	#Test models
	pred = model.predict(X_test)

	#Generate Metrics to model avaliation
	print(classification_report(Y_test,pred)) #General metrics os the model
	scores.append((name, cross_val_score(model, X, Y, scoring='f1', cv=kf, n_jobs=-1))) #Cross-Validation to verify possible overfitting
	model_displays[name] = plot_roc_curve(model, X_test, Y_test, ax=ax, name=name)

#Display Cross-Validation results	
for i in scores:
	print(i)

#Display ROC curve
_ = ax.set_title('ROC curve')
plt.show()
