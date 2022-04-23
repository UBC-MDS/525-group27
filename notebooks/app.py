from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
## Import any other packages that are needed

app = Flask(__name__)

# 1. Load your model here
model = joblib.load('model.joblib')

# 2. Define a prediction function
def return_prediction(data):

    # format input_data here so that you can pass it to model.predict()
    X_cols = ['ACCESS-CM2',
            'ACCESS-ESM1-5',
            'AWI-ESM-1-1-LR',
            'BCC-CSM2-MR',
            'BCC-ESM1',
            'CMCC-CM2-HR4',
            'CMCC-CM2-SR5',
            'CMCC-ESM2',
            'CanESM5',
            'EC-Earth3-Veg-LR',
            'FGOALS-g3',
            'GFDL-CM4',
            'INM-CM4-8',
            'INM-CM5-0',
            'KIOST-ESM',
            'MIROC6',
            'MPI-ESM-1-2-HAM',
            'MPI-ESM1-2-HR',
            'MPI-ESM1-2-LR',
            'MRI-ESM2-0',
            'NESM3',
            'NorESM2-LM',
            'NorESM2-MM',
            'SAM0-UNICON',
            'TaiESM1']
    data_df = pd.DataFrame(data = np.array(data).reshape(-1,25), columns = X_cols)
    return model.predict(data_df)

# 3. Set up home page using basic html
@app.route("/")
def index():
    # feel free to customize this if you like
    return """
    <h1>Welcome to our rain prediction service</h1>
    To use this service, make a JSON post request to the /predict url with 25 climate model outputs.
    """

# 4. define a new route which will accept POST requests and return model predictions
@app.route('/predict', methods=['POST'])
def rainfall_prediction():
    content = request.json  # this extracts the JSON content we sent
    prediction = return_prediction(content)[0]
    results = {"Input": content,
               "Output": f"The predicted rainfall is {prediction} mm."}  
    return jsonify(results)
