# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 19:51:53 2021

@author: junseonglee

Code for number recognition from MNIST database
using snorkel with simple label functions defined using train images

Written by refering to tensorflow mnist tutorial & 
https://www.tensorflow.org/tutorials?hl=ko

snorkel tutorial from
https://www.snorkel.org/use-cases/
"""
#Required packages
#!pip install tensorflow
#!pip install snorkel

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from tensorflow.keras import datasets, layers, models,Sequential,utils
from snorkel.labeling import labeling_function
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import categorical_crossentropy

print ('Start')
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()


train = 200 #max 60000
test=20     #max 10000
#Number of labeling functions defined using training data set
n_labeling_func=100
#Labeling function threshold
threshold=0.90

train_images=train_images[0:train,:,:]
train_labels=train_labels[0:train]

test_images=test_images[0:test,:,:]
test_labels=test_labels[0:test]
Y_test=test_labels

train_images = train_images.reshape((train, 784))
test_images = test_images.reshape((test, 784))

ABSTAIN=-1


#%%Definition of labeling function 

l=[]
l.append('import numpy as np\n')
l.append('from tensorflow.keras import datasets, layers, models,Sequential,utils\n')
l.append('from tensorflow.keras.datasets import mnist\n')
l.append('(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()\n')
l.append('train_images = train_images.reshape((60000, 784))\n')
l.append('test_images = test_images.reshape((10000, 784))\n')



l.append('from snorkel.labeling import labeling_function\n\n')


l.append('threshold=%.10f\n'%threshold)
l.append('ABSTAIN=-1\n')

for i in range(0, n_labeling_func):
    l.append('@labeling_function()\n')
    l.append('def label_function%i(x):\n'%(i))
    l.append('    label=int(train_labels[%d])\n'%i)
    l.append('    train=(train_images[%d])\n'%i)
    l.append('    relation=(2*sum(np.multiply(x,train))/(sum(np.multiply(x,x))+sum(np.multiply(train,train))))\n')                   
    l.append('    return label if (relation>threshold) else ABSTAIN\n\n')

l.append('lfs=[]\n')

for i in range(0, n_labeling_func):
    l.append("lfs.append(label_function%d)\n"%i)

f = open("label_functions.py", "w")
nl=np.size(l)
for i in range(0, nl):
    f.write(l[i])
f.close()

#%% Train labeling function generative model
import label_functions as lfc
from snorkel.labeling import LFApplier
from snorkel.labeling import LFAnalysis


lfs=lfc.lfs


applier = LFApplier(lfs=lfs)
L_train = applier.apply(train_images)
L_valid = applier.apply(test_images)
Y_valid = test_labels
LFAnalysis(L=L_valid, lfs=lfs).lf_summary(Y_valid)


from snorkel.labeling.model import LabelModel

label_model = LabelModel(cardinality=10, verbose=True)
label_model.fit(L_train, seed=123, lr=0.01, log_freq=10, n_epochs=10)


print(label_model.score(L_valid, Y_valid, metrics=["f1_micro"]))


probs_train= label_model.predict_proba(L=L_train)
probs_test= label_model.predict_proba(L=L_valid)

#%% Discriminative model for number recognition
y_train=np.zeros((train,10))
y_test=np.zeros((test,10))

for i in range(0, train):
    tmp=train_labels[i]
    y_train[i,tmp]=1

for i in range(0, test):
    tmp=test_labels[i]
    y_test[i,tmp]=1

# Create the model
model = models.Sequential()
model.add(tf.keras.layers.Dense(40, activation='relu'))
model.add(tf.keras.layers.Dense(30, activation='relu'))
model.add(tf.keras.layers.Dense(20, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

#print(model.summary())


# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Fit data to model
model.fit(probs_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(probs_test,  y_test, verbose=2)


print('\nGenerative model Accuracy:', test_acc)


    
from scipy.stats import mode
def majority_accuracy(lfs, label):
    ns, nl=np.shape(lfs)
    counter=0
    for i in range(0, ns):
        bindo=np.zeros(10)
        for j in range(0, nl):
            if(lfs[i,j]!=-1):
                bindo[lfs[i,j]]+=1
        if(label[i]==bindo[np.argmax(bindo)]):
            counter+=1        
    accuracy=counter/ns
    return accuracy

print('Majority model accurcay: %f' %(majority_accuracy(L_valid,test_labels)))

