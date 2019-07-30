# Requirements
* Tensorflow = 1.12
* Keras = 2.x
* OpenCV 3
* PIL
* Flask
* Numpy

#Test
* curl -X POST -F image=@test1.jpg 'http://localhost:5000/predict'
* returns Json response with predictions and sucess message
