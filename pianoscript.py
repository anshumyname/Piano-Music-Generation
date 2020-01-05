print("\n\n=========================START====================\n")
print("\nLoading libraries.......")
      
import numpy as np
import pandas as pd
import keras
import csv
import os
from keras.utils.np_utils import to_categorical
from keras.layers import Dense,Dropout,Activation,LSTM
from keras.models import Model,Sequential
from music21 import stream
from music21 import converter, instrument,note,chord
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

print("...Loaded...")
print("\n\nEnter Piano notes with keys as A-G then the octave number 1-6")
print("For example .. A6, means 'A' note of 6th octave")
print("For multiple notes entry separate them by ','.. like {A6,B7,D2,G5}")


seq_len=100
input_line= input()
input_segment= input_line.split(",")
input_seg= input_segment
while len(input_seg)<100:
        input_seg+=input_seg
input_seg= np.array(input_seg)
input_seg= input_seg[:seq_len]

print("Here your input piano tune..........\n\n")

ofset=0
input_notes=[]
for nt in input_segment:
    newnote= note.Note(nt)
    newnote.offset= ofset
    newnote.storedInstrument= instrument.Piano()
    ofset+=0.5
    input_notes.append(newnote)

inpmidi=stream.Stream(input_notes)
inpmidi.write('midi',fp='C:/Users/user/pianofiles/input.mid')
os.startfile('C:/Users/user/pianofiles/input.mid')

#Creating note->int dictionary from csv file notes.csv
notes2int={}

with open('C:/Users/user/pianofiles/notes.csv') as f:
    reader= csv.reader(f)
    for line in reader:
        notes2int[line[1]]=int(line[0])

int2note=dict((notes2int[x],x) for x in notes2int)

#Prediction model
input_ex= [notes2int[x] for x in input_seg]


def make_model():
    model = Sequential()
    model.add(LSTM(512,input_shape=([seq_len,1]),return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dense(256))#activation='relu')
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(notes2int)))#,activation='softmax')
    model.add(Activation('softmax'))
    return model
model = make_model()
#model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.load_weights('../pianofiles/weights3layermodel.h5')

pred_out=[]
pattern= np.array(input_ex)/len(notes2int)

print("\n\nPredicting notes.. Hold on..")
for ni in range(100):
    #print("Prediction number :",ni+1)
    predi=np.reshape(pattern,(1,len(pattern),1))
    #print(predi)
    #predi=predi/len(note2int)
    
    prediction= model.predict(predi)
    i=np.argmax(prediction)
    res=int2note[i]
    pred_out.append(res)
    
    pattern=np.append(pattern,i/len(notes2int))
    pattern=pattern[1:]

offset = 0

output_notes = input_notes
                         

for pattern in pred_out:
    if(('.' in pattern) or pattern.isdigit()):
        #print("Chord ->",pattern)
        notesinchord=pattern.split(".")
        notes=[]
        for cur in notesinchord:
            newnote=note.Note(int(cur))
            newnote.storedInstrument=instrument.Piano()
            notes.append(newnote)
        newchord= chord.Chord(notes)
        newchord.offset= offset
        output_notes.append(newchord)
    else:
        newnote=note.Note(pattern)
        newnote.offset = offset
        newnote.storedInstrument = instrument.Piano()
        output_notes.append(newnote)
    
    offset+=1.0

midistream= stream.Stream(output_notes)
midistream.write('midi',fp='C:/Users/user/pianofiles/output.mid')

print("\n\nHere's the melody produced from your input.....")
      
os.startfile('C:/Users/user/pianofiles/output.mid')
os.closefile('C:/Users/user/pianofiles/output.mid')

print("=======================END=======================")

    
        
    
    

