# keras_server.py 

# Python program to expose a ML model as flask REST API 

# import the necessary modules 
from keras.applications import ResNet50 # pre-built CNN Model 
from keras.preprocessing.image import img_to_array 
from keras.models import load_model
from keras.applications import imagenet_utils 
import tensorflow as tf 
from PIL import Image 
import numpy as np 
import flask 
import cv2
import io 

# Create Flask application and initialize Keras model 
app = flask.Flask(__name__) 
model = None
IMG_SIZE = 50  # 50 in txt-based

CATEGORIES = ["Oily", "Unkwown"]
def prepare(filepath):
    IMG_SIZE = 50  # 50 in txt-based
    new_array = cv2.resize(filepath, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# Function to Load the model 
def load_models(): 

	# global variables, to be used in another function 
	global model	 
	model = load_model('Skin.hdf5')
	global graph 
	graph = tf.get_default_graph() 

# Every ML/DL model has a specific format 
# of taking input. Before we can predict on 
# the input image, we first need to preprocess it. 
def prepare_image(image, target): 
     
    image = image.convert("L") 

    # Resize the image to the target dimensions 
    image = image.resize(target) 

    # PIL Image to Numpy array 
    image = img_to_array(image) 

    # Expand the shape of an array, 
    # as required by the Model 
    image = np.expand_dims(image, axis = 0) 


    image = image.reshape(-1,50,50,1)   

    # return the processed image 
    return image 

# Now, we can predict the results. 
@app.route("/predict", methods =["POST"]) 
def predict(): 
    data = {} # dictionary to store result 
    data["success"] = False

    # Check if image was properly sent to our endpoint 
    if flask.request.method == "POST": 
        if flask.request.files.get("image"): 
            image = flask.request.files["image"].read() 
            image = Image.open(io.BytesIO(image)) 

            image = image.convert("L") 
            image = image.resize((IMG_SIZE,IMG_SIZE))
            image = np.array(image)
            image = image.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
                        
        # Predict ! global preds, results 
            with graph.as_default(): 
                prediction = model.predict(image) 
                results = CATEGORIES[int(prediction[0][0])]
                data["predictions"] = results 


            data["success"] = True

    # return JSON response 
    return flask.jsonify(data) 



if __name__ == "__main__": 
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started")) 
    load_models() 
    app.run() 
