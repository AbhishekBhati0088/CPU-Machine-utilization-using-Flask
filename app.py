# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:00:26 2023

@author: I4628
"""

from flask import Flask,render_template,request
import pickle
import numpy as np



app = Flask(__name__)



model = pickle.load(open("C:/Users/I4628/Abhishek/final_rf.pkl","rb"))

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def home():
    
    min = request.form['a']
    max =request.form['b']
    average = request.form['c']
    cpu = request.form['d']
    diskfree = request.form['e']
    memory = request.form['f']
    array = np.array([[min,max,average,cpu,diskfree,memory]])
    pred = model.predict(array)
    return render_template("result.html",data=pred)


if __name__=="__main__":
    app.run(debug=True)