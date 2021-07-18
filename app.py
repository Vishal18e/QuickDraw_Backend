import os
from flask import Flask,request
import json
from flask_cors import CORS
import base64
import Assests


app = Flask(__name__)
cors = CORS(app)
datasetPath ='data'


@app.route('/api/upload_canvas',methods=['POST'])
def upload_canvas():
    data = json.loads(request.data.decode('utf-8'))
    image_data =data['image'].split(',')[1].encode('utf-8')
    fileName =data['filename']
    className =data['className']
    os.makedirs(f'{datasetPath}/{className}/image',exist_ok=True)
    with open(f'{datasetPath}/{className}/image/{fileName}',"wb")as fh:
        fh.write(base64.decodebytes(image_data))

    return "Got the Image"


@app.route('/api/play',methods=['POST'])
def play():

    data = json.loads(request.data.decode('utf-8'))
    image_data =data['image'].split(',')[1].encode('utf-8')
    fileName =data['filename']

    className ='temp';

    os.makedirs(f'{datasetPath}/{className}/image',exist_ok=True)
    with open(f'{datasetPath}/{className}/image/{fileName}',"wb")as fh:
        fh.write(base64.decodebytes(image_data))
    className=Assests.test((datasetPath+'/'+className+'/'+'image'+'/'+fileName))
    return className

# if __name__ =="main":
#     app.run(debug=True)