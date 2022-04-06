import sys
import os
import glob
import re
import numpy as np

# Keras
#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
#from keras.models import load_model
#from keras.preprocessing import image

# classifier = tf.keras.models.load_model("/content/SolarPanelDefectPrediction.h5")

Prediction=['Pannel is Defective','Pannel is Non Defective']

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
import numpy as np
from glob import glob
#import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras.preprocessing import image



from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image

print('Model loaded. Check http://127.0.0.1:5000/')

app = Flask(__name__)



model = tf.keras.models.load_model("SolarPanelDefectPrediction.h5")

#model.make_predict_function()

def predict_label(img_path):
	#i = image.load_img(img_path, target_size=(224,224))
	#i = image.img_to_array(i)/255.0
	#i = i.reshape(1, 100,100,3)
	#p = model.predict_classes(i)
	#return dic[p[0]]
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x=x/255
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')
    print(img_path)
    preds = model.predict(x)
    for i in range(len(Prediction)):
      if preds[0][i] == 1:
        print(Prediction[i])
        result=Prediction[i]
    print(result)
    return result

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
