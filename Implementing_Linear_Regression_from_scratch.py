#!/usr/bin/env python
# coding: utf-8

# Linear Regression:
# 
# **Y = wX + b**
# 
# Y --> Dependent Variable
# 
# X --> Independent Variable
# 
# w --> weight
# 
# b --> bias

# **Gradient Descent:**
# 
# Gradient Descent is an optimization algorithm used for minimizing the loss function in various machine learning algorithms. It is used for updating the parameters of the learning model.
# 
# w  =  w - α*dw
# 
# b  =  b - α*db

# Importing the Dependencies

# In[5]:


# Importing numpy library
import numpy as np


# **Linear Regression**

# In[6]:


class Linear_Regression():

   def __init__( self, learning_rate, no_of_iterations ) :

        self.learning_rate = learning_rate

        self.no_of_iterations = no_of_iterations

    # fit function to train the model

   def fit( self, X, Y ) :

        # no_of_training_examples, no_of_features

        self.m, self.n = X.shape

        # initiating the weight and bias

        self.w = np.zeros( self.n )

        self.b = 0

        self.X = X

        self.Y = Y


        # implementing Gradient Descent for Optimization

        for i in range( self.no_of_iterations ) :

            self.update_weights()



    # function to update weights in gradient descent

   def update_weights( self ) :

        Y_prediction = self.predict( self.X )

        # calculate gradients

        dw = - ( 2 * ( self.X.T ).dot( self.Y - Y_prediction )  ) / self.m

        db = - 2 * np.sum( self.Y - Y_prediction ) / self.m

        # updating the weights

        self.w = self.w - self.learning_rate * dw

        self.b = self.b - self.learning_rate * db


    # Line function for prediction:

   def predict( self, X ) :

        return X.dot( self.w ) + self.b


# Using Linear Regression model for Prediction

# In[7]:


# importing the dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Data Pre-Processing

# In[8]:


# loading the data from csv file to a pandas dataframe

salary_data = pd.read_csv('/content/salary_data.csv')


# In[9]:


# printing the first 5 columns of the dataframe
salary_data.head()


# In[10]:


# last 5 rows of the dataframe
salary_data.tail()


# In[11]:


# number of rows & columns in the dataframe
salary_data.shape


# In[12]:


# checking for missing values
salary_data.isnull().sum()


# Splitting the feature & target

# In[13]:


X = salary_data.iloc[:,:-1].values
Y = salary_data.iloc[:,1].values


# In[14]:


print(X)


# In[15]:


print(Y)


# Splitting the dataset into training & test data

# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state = 2)


# Training the Linear Regression model

# In[17]:


model = Linear_Regression(learning_rate = 0.02, no_of_iterations=1000)


# In[18]:


model.fit(X_train, Y_train)


# In[19]:


# printing the parameter values ( weights & bias)

print('weight = ', model.w[0])
print('bias = ', model.b)


# y = 9514(x) + 23697
# 
# 
# salary = 9514(experience) + 23697

# Predict the salary value for test data

# In[20]:


test_data_prediction = model.predict(X_test)


# In[21]:


print(test_data_prediction)


# Visualizing the predicted values & actual Values

# In[22]:


plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, test_data_prediction, color='blue')
plt.xlabel(' Work Experience')
plt.ylabel('Salary')
plt.title(' Salary vs Experience')
plt.show()

