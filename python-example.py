#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[122]:


df = pd.read_csv('WA_Fn-UseC_-Accounts-Receivable.csv')
lb = LabelEncoder()
df['Disputed'] = lb.fit_transform(df['Disputed'].astype('str'))
df['customerID'] = lb.fit_transform(df['customerID'].astype('str'))
df['PaperlessBill'] = lb.fit_transform(df['PaperlessBill'].astype('str'))

df['PaperlessDate'] = pd.to_datetime(df['PaperlessDate'])
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['DueDate'] = pd.to_datetime(df['DueDate'])
df['SettledDate'] = pd.to_datetime(df['SettledDate'])


df['DueDate'] = [datetime.datetime.timestamp(df['DueDate'].loc[0]) for i in range(0,len(df))]
df['PaperlessDate'] = [datetime.datetime.timestamp(df['PaperlessDate'].loc[0]) for i in range(0,len(df))]
df['InvoiceDate'] = [datetime.datetime.timestamp(df['InvoiceDate'].loc[0]) for i in range(0,len(df))]
df['SettledDate'] = [datetime.datetime.timestamp(df['SettledDate'].loc[0]) for i in range(0,len(df))]


# In[125]:


# ?creating the network architecture 
x_train, x_test, y_train, y_test = train_test_split(df.drop('DaysLate', axis=1), df['DaysLate'])
model = tf.keras.models.Sequential([
                                    # Dense is the type of layer, its simple layer of neurons, which we are using here, the others are known as Conv2d and maxpooling2d
                                    # They are used for images
                                    # It contains 32 neurons and activation function being used is 'relu', input dimension is 1 as the input column is only 1
                                    tf.keras.layers.Dense(32, input_dim=11, activation='relu'),
                                    # It contains 128 neurons and activation function being used is 'relu'
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    # It contains 1024 neurons and activation function being used is 'relu'
                                    tf.keras.layers.Dense(1024, activation='relu'),
                                    tf.keras.layers.Dense(256, activation='relu'),
                            
                                    tf.keras.layers.Dense(64,activation='relu'),
                                    # output layer 1, only 1 neuron is being used because we only want to predict only 1 value 
                                    tf.keras.layers.Dense(1, activation='linear')
])
# RMSprop is the optimizer with learning rate is 0.001
opt=tf.keras.optimizers.RMSprop(learning_rate=0.001)
# some parameters will be set in this line
# Prescribing loss function is mean_squared_error which the neural network will try to minimize, metric is also mean_squared_error against
# which model will be evaluated
model.compile(loss='mean_squared_error', optimizer=opt, metrics = ['mean_squared_error'])
# Here the data is passed through the network, epochs defines number of times data will be passed, validation_data
# is passed to test the network on unseen data and batch_data defines after how much time the weight will be adjusted
history = model.fit(x_train, y_train, epochs=500, batch_size=32, validation_data =(x_test, y_test))
