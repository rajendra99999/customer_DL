#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import numpy as np
from flask import Flask,request,render_template,jsonify,url_for
from keras.models import load_model
model=load_model('model.h5')
lt=WordNetLemmatizer()

# In[2]:
kv={'highly satisfied':5,'satisfied':4,'moderatly satisfied':3,'not satisified ':2,'highly dissatified':1}

app=Flask(__name__)



# In[3]:


@app.route('/')
def home():
    return render_template('result.html')


# In[4]:


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sen=str(request.form['comment'])
        
        review1=re.sub('[^a-zA-Z]',' ',sen)
        review1=review1.lower()
        review1=review1.split()
        review1=[lt.lemmatize(word) for word in review1 if not word in stopwords.words('english')]
        review1=' '.join(review1)
        
        onehot_repx=[one_hot(review1,voc_size)]
        
        emd_x=pad_sequences(onehot_repx,padding='pre',maxlen=100)
        
        x_input=np.array(emd_x)
        
        y_out=model.predict_classes(x_input)
        
        for i,j in kv.items():
            if y_out==j:
                output=i
    
       
        
        return render_template('result.html', prediction_text='your customer is {}'.format(output))

if __name__=='__main__':
        app.run(debug=True)
    


# In[ ]:




