# -*- coding: utf-8 -*-
# This is the second tutorial of Math of intelligence from Siraj
# For this week we are going to implement linear regression for google stock price prediction
# We will predict the stock price for the next day

# Importing the dependency
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import  train_test_split

# import the datasets
data_path='Google_Stock_Price_Train.csv'
dataset= pd.read_csv(data_path)
X_train=dataset.iloc[0:len(dataset)-1,[1]].values
Y_train=dataset.iloc[1:len(dataset),[1]].values

# Split training and test set
X_train,X_test, Y_train ,Y_test = train_test_split(X_train,Y_train,test_size=0.3,random_state=1)
# feature scaling
scaler = MinMaxScaler()
X_train =scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)

def mss(theta,data,target):
    num_instance = len(data)
    error=0
    for i in range(num_instance):
        predicted=theta[0]+theta[1]*data[i]
        error+=(target[i]-predicted)**2
    return error/num_instance

def compute_jacobian(theta, data , target ):
    jaccobian=np.zeros(len(theta))
    for i in range(len(data)):
        jaccobian[0] +=-(target[i]-theta[0]-theta[1]*data[i])
        jaccobian[1] += -(target[i] - theta[0] - theta[1] * data[i])*data[i]
    return jaccobian

def compute_hessien(theta,data,target):
    hessien =np.zeros([len(theta),len(theta)])
    for i in range(len(data)):
        hessien[0,0]+=1
        hessien[0,1]+=data[i]
        hessien[1,0]+=data[i]
        hessien[1,1]+=data[i]**2
    return hessien
def newton_optimization(theta,data,target,num_iteration = 300,eps = 1e-5):

    for i in range(num_iteration):
        jaccobian = compute_jacobian(theta,data,target)
        hessien = compute_hessien(theta,data,target)
        theta = np.array(np.subtract(theta, np.dot(np.linalg.inv(hessien), jaccobian)))
        if np.abs(mss(theta,data,target))<eps:
            break
        print('Optimal m is %.2f and Optimal b is %.2f' % (theta[0], theta[1]))
    return theta

def make_prediction(theta,data):
    num_instance = len(data)
    predicted = np.zeros(num_instance)
    for i in range(num_instance):
        predicted[i] = theta[0]+theta[1]*data[i]
    return predicted

def ploting_the_decision(theta,data,target):
    plt.plot(data, target, 'bo') #First plots the data points
    plt.plot(data,  theta[0]+theta[1]* data) #Plot the line.
    plt.title("Best line.")
    plt.show() #shows the graph.

if __name__=='__main__':
    theta = np.array([0.001, 0.002])
    theta = newton_optimization(theta, X_train, Y_train)
    predicted = make_prediction(theta,X_test)
    error = mss(theta,X_test,Y_train)
    print(error)
    ploting_the_decision(theta, X_train, Y_train)

