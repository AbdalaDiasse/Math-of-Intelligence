# This is the second tutorial of Math of intelligence from Siraj
# For this week we are going to implement logistic regression for churn modeling
# The dispose user information from a bank and we want predict weather or not the costumer will
# leave the bank or not

# Importing the dependency
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import  train_test_split
from math import exp,log
from sklearn.metrics import  accuracy_score

# import the datasets
data_path='C:/Users/Abdoullahi/Math-of-Intelligence/week 2：Second order newton optimization' \
          '/Churn_Modelling.csv'
dataset= pd.read_csv(data_path)
X_train=dataset.iloc[:,3:13].values
Y_train=dataset.iloc[:,13].values
X_train=X_train[:1000,:]
Y_train=Y_train[:1000]

# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
X_train[:, 1] = labelencoder_X_1.fit_transform(X_train[:, 1])
labelencoder_X_2 = LabelEncoder()
X_train[:, 2] = labelencoder_X_2.fit_transform(X_train[:, 2])

# Split training and test set
X_train,X_test, Y_train ,Y_test = train_test_split(X_train,Y_train,test_size=0.3,random_state=1)
# feature scaling
scaler = MinMaxScaler()
X_train =scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)

def sigmoid(theta,data_i):
    num_param = len(theta)
    the_sum = 0
    for i in range(num_param):
        the_sum =the_sum + theta[i]*data_i[i]
    return 1/(1-exp(the_sum))

def compute_error(theta,data,target):
    error =0
    num_instance=len(data)
    num_param= len(theta)
    for i in range(num_instance):
        sum_1=0
        for j in range(num_param):
            sum_1 += theta[j]*data[i,j]
        error += (target[i]*sum_1 + log(1+exp(-sum_1)))
    return error

def gradient_decent_step(theta, data , target , rate):
    num_instances = len(data)
    num_param = len(theta)
    for j in range(num_param):
        dr_theta_j = 0
        for i in range(num_instances):
            dr_theta_j += (sigmoid(theta, data[i, :])-target[i])*data[i, j]
        theta[j] = theta[j] + rate * dr_theta_j
    return theta


def gradient_decent_n_step(theta,data,target,rate,epochs):
    error = np.zeros(epochs)
    for i in range(epochs):
        error[i] = compute_error(theta,data,target)
        theta = gradient_decent_step(theta,data,target,rate)
    return theta,error

def make_prediction(theta,data):
    num_instance=len(data)
    predicted=np.zeros(num_instance)
    for i in range(num_instance):
        predicted[i] = sigmoid(theta, data[i, :])
    for i in range(num_instance):
        if (predicted[i]>=0.5):
            predicted[i]=1
        else:
            predicted[i]=0
    return predicted

epochs = 500
rate = 0.00001
theta=np.array([0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001])
theta ,error=gradient_decent_n_step(theta, X_train, Y_train, rate,epochs)
predicted = make_prediction(theta,X_test)
accuracy = accuracy_score(Y_test,predicted)

# Ploting the
iterations=range(len(errors))
plt.plot(iterations,errors)
plt.ylabel('errors')
plt.xlabel('Iterations')
plt.show()