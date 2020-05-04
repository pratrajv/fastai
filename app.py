from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import os.path
import fetch_file as ff


# Import fast.ai Library
from fastai.vision import *
#from torch.vision import *

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)


path = Path("path")
classes = ['black', 'grizzly', 'teddys']
data2 = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = create_cnn(data2, models.resnet34)
file_id = '1iHHb6vBiJTwzDKb6rFJjwo2NT3p4Q7UC'
dir = os.path.dirname(os.path.abspath(__file__)) + '/path/models/stage-2.pth'
ff.download_file_from_google_drive(file_id, dir)
learn.load('stage-2')



def model_predict(img_path):
    """
       model_predict will return the preprocessed image
    """
   
    img = open_image(img_path)
    pred_class,pred_idx,outputs = learn.predict(img)
    return pred_class
    




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)
        return preds
    return None


if __name__ == '__main__':
    
    app.run()


