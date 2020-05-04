"""
Backend script to give predictions and train speech emotion recognition

Author: Claudio Fanconi
Email: claudio.fanconi@outlook.com
"""

# Import libraries
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import preprocessing
import pickle
from scipy.io import wavfile
import librosa
import os

app = Flask(__name__)

# Broadcast
HOSTNAME = "0.0.0.0"
# HTTP Port
PORT = 5000
DATA_PATH = "data/"
WAV_PATH = "wav/"
LABEL_INDEX = 5

german_to_labels = {'W' : 0, 'L' : 1, 'E' : 2,
                    'A' : 3, 'F' : 4, 'T' : 5, 'N' : 6}

label_to_description = {0 : 'Anger',  1 : 'Boredom', 2 : 'Disgust',
                        3 : 'Fear', 4 : 'Happiness', 5 : 'Sadness', 6 : 'Neutral'}

# Source: https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/
def extract_feature(X, sample_rate, 
                    mfcc=True, chroma= False, mel= False):
    """
    This function creates features, to be used for speech emotion recognition
    params:
        X: the frequencies of the speech
        sample_rate: the rate of the frequencies
        mfcc (default = True): Mel-frequency cepstrum coefficients - which bins the various 
                               frequencies into a heatmap.
        chroma (default = False): Additional Chroma transformation
        mel (default = False): Addtional mel transormation
    return:
        Features based on the MFCC transformation
    """
    X = X.astype("float32")
    if chroma:
        stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

# Get a list of all the wav file names
list_of_files = sorted(os.listdir(DATA_PATH + WAV_PATH))
map_of_files = dict(zip(list_of_files, range(len(list_of_files))))

X = []
y = []
print("Loading in the Data...")
for f in list_of_files:
    # Get the label
    y.append(german_to_labels[f[LABEL_INDEX]])

    # Get the features
    wav = (wavfile.read(DATA_PATH + WAV_PATH + f))
    sample_rate = wav[0]
    vector = wav[1]
    X.append(extract_feature(vector, sample_rate, mfcc = True, chroma=True, mel=True))
    
X = np.array(X)
X = preprocessing.scale(X)
y = np.array(y)
print("Done.\n")

print("Loading Classifier from Disk...")
# load the model from disk
model = SVC(C = 10, class_weight="balanced", probability=True, gamma = "auto")
print("Done.\n<k")


# root
@app.route("/")
def index():
    """
    this is a root dir of my server
    return:
        str where the user is greeted
    """
    return "hello there %s" % request.remote_addr


# GET
@app.route("/train")
def train():
    """
    This endpoint is used to retrain the model
    params:
        None
    return:
        None
    """
    print("Evaluating Model...")
    scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=4,
                                                             shuffle=True,
                                                             random_state=42), scoring="f1_macro")
    print("Fitting Model...")
    model.fit(X,y)
    mean_cv_score = np.array(scores).mean()
    std_cv_score = np.array(scores).std()
    
    response = {
        "STATUS" : "FITTED", 
        "F1_MACRO_MEAN" : mean_cv_score, 
        "F1_MACRO_STD" : std_cv_score
        }
    return jsonify(response)

# POST
@app.route("/predict", methods=["POST"])
def predict():
    """
    Predicts the emotion of a speech sample. The speech file is automatically extracted from the POST request
    return:
        json response containing the prediciton
    """
    json = request.get_json()
    filename = str(json["FILENAME"])

    if filename not in map_of_files:
        response = {"STATUS" : "FAILED", "ERROR" : "DATA NOT FOUND"}
        return jsonify(response)

    else:
        # Get the right features from the dataset
        index = map_of_files[filename]
        X_single = X[index].reshape(1,-1)

        # Predict Model
        try:
            y_pred = model.predict(X_single)
            y_proba = model.predict_proba(X_single)
            emotion = label_to_description[y_pred[0]]

        except:
            response = {
                "STATUS" : "FAILED",
                "ERROR" : "MODEL NOT FITTED YET"
            }
            return jsonify(response)

        # Build response:
        response = {
            "FILENAME" : filename,
            "EMOTION" : emotion,
            "PROBABILITY" : y_proba.max()
        }
        return jsonify(response)

if __name__ == "__main__":
    app.run(host=HOSTNAME, port=PORT)
