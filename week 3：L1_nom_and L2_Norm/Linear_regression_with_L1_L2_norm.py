# -*- coding: utf-8 -*-
# This is the  third tutorial of Math of intelligence from Siraj
# For this week we are going to implement linear regression for google stock prediction
# We dispose the stock price and we need to predict the stock price for the next week

# Importing the dependency
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import  train_test_split
from math import sqrt
from sklearn.metrics import  accuracy_score

# import the datasets
data_path = 'C:/Users/Abdoullahi/Desktop/udemy/PART 3. RECURRENT NEURAL NETWORKS (RNN)/Recurrent_Neural_Networks' \
          '/Google_Stock_Price_Train.csv'
dataset = pd.read_csv(data_path)
dataset=dataset.iloc[:,1:2].values
# feature scaling
scaler = MinMaxScaler()
dataset =scaler.fit_transform(dataset)


# Creating Features with 7 timesteps
num_instances = len(dataset)
X_train = []
Y_train = []
for i in range(num_instances-7):
    X_train.append(dataset[i:i+7,0])
    Y_train.append(dataset[[i+7], 0])
X_train=np.array(X_train)
Y_train=np.array(Y_train)

# Split training and test set
X_train,X_test, Y_train ,Y_test = train_test_split(X_train,Y_train,test_size=0.3,random_state=1)

def rmss(theta,data,target):
    mss =0
    num_instance=len(data)
    for i in range(num_instance):
        predicted = np.dot(data[i,:],theta[1:])
        mss += (predicted-target[i])**2/float(num_instances)
    return sqrt(mss)

def gradient_decent_step_L2_norm(theta, data , target , rate,alpha):
    num_param = len(theta)
    derivative_theta_i=0
    data = np.concatenate((np.ones([len(data),1]),data),axis=1)
    theta=theta.reshape([len(theta),1])
    error = np.subtract(np.dot(data[:,:],theta),target)
    for i in range(num_param):
        derivative_theta_i += (np.dot(data[:, i],error) + alpha*theta[i])/float(len(data))
        theta[i] = theta[i] - rate * derivative_theta_i
    return theta

def gradient_decent_n_step(theta,data,target,rate,epochs,alpha):
    error = np.zeros(epochs)
    for i in range(epochs):
        error[i] = rmss(theta,data,target)
        theta = gradient_decent_step_L2_norm(theta,data,target,rate,alpha)
    return theta,error
def gradient_decent_n_step_n_alpha(theta,data,target,rate,epochs,alpha):
    error_n = []
    theta_n = []
    for i in range(len(alpha)):
        print(i)
        theta_i,error_i = gradient_decent_n_step(theta,data,target,rate,epochs,alpha[i])
        error_n.append(error_i)
        theta_n.append(theta_i)
    return error_n,theta_n

def make_prediction(theta,data):
    predicted=np.dot(data,theta[1:,])
    return predicted

def ploting_error(error):
    for i in range(len(error)):
        iterations = range(len(error[i]))
        plt.plot(iterations, error[i])
        plt.ylabel('errors')
        plt.xlabel('Iterations')
    plt.show()

epochs = 500
rate = 1
theta = np.zeros(8)
alpha=np.array([0.001,0.01,0.1,1,10,100,100])
error_n,theta_n = gradient_decent_n_step_n_alpha(theta, X_train, Y_train, rate, epochs, alpha)
predicted = make_prediction(theta,X_test)
ploting_error(error_n)

