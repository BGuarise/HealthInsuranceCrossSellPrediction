import csv
import json
import numpy as np 
import pandas as pd
import joblib

df_test = pd.read_csv("Insurance_data/test.csv")
jsonfile = open('Insurance_data/test.json', 'w')

json.dump(df_test.to_json(orient='records'), jsonfile)
