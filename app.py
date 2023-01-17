#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:26:40 2022

@author: macbook
"""
import pickle
import re
import os
import numpy as np
import tensorflow.keras as keras
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Bio import SeqIO

app = Flask(__name__, template_folder='templates')
app._static_folder = os.path.abspath("templates/static/")

model = keras.models.load_model("model_gru.h5")

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def build_kmers(sequences, ksize):
    kmers = []
    n_kmers = len(sequences) - ksize + 1

    for i in range(n_kmers):
        kmer = sequences[i:i + ksize]
        kmers.append(kmer)

    return kmers

def inferencing(input1):
    input1 = re.sub(r"[\n\t\s]*", "", input1)
    kmer_out = []
    if len(input1)>=5000:
        kmer_out.append(build_kmers(input1[:5000], 6))
    else:
        kmer_out.append(build_kmers(input1, 6))

    out = tokenizer.texts_to_sequences(kmer_out)
    out = pad_sequences(out, padding='post',
                                 truncating='post', maxlen=4995)
    out = np.asarray(out).astype(np.float32)
    hasil=model.predict(out)
    print(hasil)
    hasil = np.argmax(hasil)
    return hasil

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # if user input text
        message = request.form['message']
        f = request.files['file']
        print("filename ", f.filename)
        if message != '':
            my_prediction = inferencing(message)      
        elif f.filename.endswith('.fasta'):        
        # if user input file                    
            f.save('upload/'+ f.filename)
            message = str(SeqIO.read('upload/'+f.filename, "fasta").seq)
            my_prediction = inferencing(message) 
        else:
            message=''
            my_prediction=99
            

        return render_template('result.html',prediction = my_prediction, message=message)   


if __name__=="__main__":
    app.run("0.0.0.0")
