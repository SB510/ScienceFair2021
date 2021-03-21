#modules
import pyaudio
import wave
import serial
import time
from scipy.fftpack import fft
from scipy.io import wavfile
from numpy import interp
import csv
from pandas import read_csv
from pickle import dump
from pickle import load
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

#Variables
arduinoSerial = serial.Serial('Com3', 9600)#the com3 may need to be changed
chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 44100  # Record at 44100 samples per second
seconds = 3
filename = "output.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio



stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

#Set the frequency domain the all files will be
timeLength = 5 #Seconds
frameRate = 1/fs
fLow = 1/timeLength
##fHigh = 1/(2*frameRate)
fHigh = 4000
fZero = []
i = fLow

while i <= fHigh:
    fZero.append(i)   
    i += fLow 






#load ML algorithms
modelADB = load(open('Tuned_ADB.sav', 'rb'))
modelSVM = load(open('Tuned_SVM.sav', 'rb'))
modelKNN = load(open('Tuned_KNN.sav', 'rb'))



    
#Definitions
def record():
    print('recording')
    stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)
    frames = []  # Initialize array to store frames
    # Store data in chunks for 5 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
##    p.terminate()
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    print('done recording')

def transform(file):
    sps, data = wavfile.read(file) # load the data
    a = data.T # this is a two channel soundtrack, I get the first track
    normalized = []
    for ele in a:
        b=(ele/2**8.)*2-1# this is 8-bit track, b is now normalized on [-1,1)
        normalized.append(b)


    c = fft(normalized) # calculate fourier transform (complex numbers list)
    d = len(c)/2  # you only need half of the fft list (real signal symmetry)
    d = int(d)
    transformed = abs(c[:(d)])
    return transformed, sps, data

def toFFTandInterpolate(file):
        transformed, fs, data = transform(file)

        
        t = len(data)/fs
        
        fLow = 1/t
        fHigh = 1/(2*(1/fs))

            
        xaxis= []
        i = fLow
        transformed = transformed[:int(4000/fLow)+1]
        while len(xaxis) < len(transformed[0]):
            xaxis.append(i)   
            i += fLow

        interpolate = interp(fZero, xaxis, transformed[0])
        return interpolate

def sendToArduino (prediction):
    if (prediction == 1): #if the value is 1
        arduinoSerial.write(b'1') #send 1
        print ("NO speach")

    if (prediction == 0): #if the value is 0
        arduinoSerial.write(b'0') #send 0
        print ("Speach is happening")












#main loop
while True:
    #collect my audiofile
    record()
    
    #perform an FFT on it and interpolate
    toPredict = [toFFTandInterpolate(filename)]
    print('prossed recording')
    #feed it into the ML algorithm for a prediction
    predictionADB = modelADB.predict(toPredict)
    predictionSVM = modelSVM.predict(toPredict)
    predictionKNN = modelKNN.predict(toPredict)
    print(predictionADB, predictionSVM, predictionKNN)
    #send that prediction to the arduino board
    sendToArduino(int(predictionKNN))








