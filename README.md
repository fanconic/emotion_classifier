# Emotion Classification Project
This project aims to build a classifier to identify different types of emotions in speech files.

## Installation
To install this project you need to run the following steps:``

Move into the source directory:
```
$ cd emotion_classifier
```

The Docker container can be build with the following command:
(This will also automatically download and unzip the data)
```
$ docker image build -t emotion_service .
```

Finally, you can run the docker container with:
```
$ docker run -p 5000:5000 -p 9000:9000 emotion_service
```

## Jupyter Notebook
The jupyter notebook willbe accessible through the link appearing in the output terminal starting like this:
```
$ http://127.0.0.1:9000/?token= ...
```


## Flask Queueries:
### /train (GET)
If you wish to train the classifier, you can do this by quering 0.0.0.0:5000/train via your webbrowser.
Furthermore, you can do this via curl in your terminal:
``` 
$ curl http://0.0.0.0:5000/train
``` 
This will give you a JSON response, which contains the STATUS, mean and standard deviation of the F1 Macro score after 4 fold cross validation. 

### /predict (POST)
If you wish to predict an existing speech file of dataset, you can send it via an HTTP POST request to the /predict end point. The JSON body of the POST request should have the following structure:
``` 
{ NAME: your_filename.wav }
``` 

The following below will give the results for 03a01Fa.wav
``` 
$ curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"FILENAME":"03a01Fa.wav"}' \
  http://0.0.0.0:5000/predict
``` 