# Piano-Music-Generation

In this project I have tried to make a piano tune predictor based on the user's input.

## Overview
Currently I haven't made a GUI of piano so i take the input in form of a string from user with following format:
A1,B2,D4,E3,G4,F6.....etc

Let me explain you here A-G will be keys of piano accordingly and 1-6 will be the
octave numbers so you have 7*6 =42 unique keys in total.

Here is a picture of 7 white keys of piano A-G. This corresponds to an octave Similar 6 of the images combine to form the complete 42 keys(white) piano
On which user has to give the input

![Distribution of A-G in on keys piano in 1 octave](https://www.researchgate.net/profile/Giovanni_De_Poli/publication/265191930/figure/fig1/AS:295924426395648@1447565289247/One-octave-in-a-piano-keyboard.png)

Then after that you will listen your input tune as well as after a few moments you will get a predicted pattern..
It may be not very good becz there are more parameters that affect a tune rather than only notes

### Libraries Requirements
pip install<br/>
  ->music21<br/>
  ->keras 2.0<br/>
  ->numpy , pandas<br/>
  ->glob and os<br/>
  
  
### Repo Fragments
Midi_files : the training examples on which the model was trained

weights3layermodel.h5: trained model weights after long computation time

Piano.ipynb:  the raw notebook  for examining the whole code in detail

Pianoscript.py:  the script for user which is for  the user testing.

Notes.csv:  all piano notes that can be predicted by the model

### How to Run?
->After installing all the requirements of code u need to just do a few things more.

->In pianoscript.py change the path according to your system.(For convenience I have mentioned where to change the paths)
    
Now open anaconda prompt or cmd:<br/>
    ->Move to folder where all files you have stored<br/>
    -> run the script by “python pianoscript.py”<br/>
    
### References
Medium : https://medium.com/@alexissa122/generating-original-classical-music-with-an-lstm-neural-network-and-attention-abf03f9ddcb4 \
Github: https://github.com/Skuldur/Classical-Piano-Composer






