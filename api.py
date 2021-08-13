from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import utils as ut

app = Flask(__name__)

@app.route('/')
def index():
	return 'Hello!\nUse /predict to get models prediction'

# Prediction Method
@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            json_ = request.json
            query = pd.read_json(json_)
            query = ut.data_transformation(query)
            prediction = list(model.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
    
    model = joblib.load("models/LR.pkl") # Load model
    print ('Model loaded')

    app.run(port=port, debug=True)