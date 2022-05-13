import os 
import json
from flask import Flask, request, jsonify, render_template, url_for

from typing import Dict
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


  # Setting up Environment Variable

# export GOOGLE_APPLICATION_CREDENTIALS="google-credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="google-credentials.json"

app = Flask(__name__,template_folder='templates/')       # Initilize App

@app.route('/')
def home():
    return render_template('titanic.html')


def predict_tabular_classification_sample(
    project: str,
    endpoint_id: str ,
    instance_dict: Dict,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com"):
    
    
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    # creating the instance schema, 
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]

    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    # Creating Endpoint Path
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )

    # Getting Respone from API
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)

    predictions = response.predictions
    
    return predictions
        

@app.route('/predict',methods=['POST'])
def predict():

    # Prepare Data Dict for Prediction
    pre_data={}
    pre_data['PassengerId']= request.form.get('PassengerId')
    pre_data['Pclass']= request.form.get('Pclass')
    pre_data['Sex']= request.form.get('Sex')
    pre_data['Age']= request.form.get('Age')
    pre_data['SibSp']= request.form.get('SibSp')
    pre_data['Parch']= request.form.get('Parch')
    pre_data['Ticket_new']= request.form.get('Ticket_new')
    pre_data['Fare']= request.form.get('Fare')
    pre_data['Embarked']= request.form.get('Embarked')

    # Getting Prediction
    prediction= predict_tabular_classification_sample(project = "auto-vision-nlp-341507",endpoint_id = "6959160935915192320",instance_dict = pre_data)
    a=[dict(i) for i in prediction]
    if a[0]['scores'][0] >= 0.5:
        result='Not Survived'
    else:
        result='Survived'
    
    return render_template('titanic.html', prediction_text='Survival prediction: {}'.format(result))

if __name__ == '__main__':
    app.run(debug=False)