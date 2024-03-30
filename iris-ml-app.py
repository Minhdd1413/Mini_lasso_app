import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

st.write("""
# Simple Iris Flower Prediction App

This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

# df = user_input_features()

# st.subheader('User Input parameters')
# st.write(df)

data = pd.read_csv('housing.csv')
st.write(data)

st.subheader('Data Description')
st.write(data.describe())

st.subheader('Unique Values')
st.write(data.nunique())

# Preprocess the data
# Drop the missing values
data = data.dropna()

# Etract the target variable
y = data['ocean_proximity'].to_numpy()
X = data.drop('ocean_proximity', axis=1).to_numpy()
st.write(X.shape, y.shape)

# Normalize the data
X = [(i - X.mean()) / X.std() for i in X]

# One hot encoding
y = pd.get_dummies(y).to_numpy()*1

# Spit the data into X and Y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=211)

model = Lasso(alpha=0.1, max_iter=1000, tol=0.0001, selection='cyclic')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

for i in range(y_pred.shape[0]):
    y_pred[i][y_pred[i] == max(y_pred[i])] = 1.0
    y_pred[i][y_pred[i] < max(y_pred[i])] = 0.0
    
#st.write(prediction)    
st.subheader('Prediction')
st.write(y_pred)

# Calculate the accuracy
accuracy = sum([1 for i in range(y_pred.shape[0]) if (y_pred[i] == y_test[i]).all()]) / y_pred.shape[0]
st.subheader('Accuracy')
st.write(accuracy)

# prediction_proba = model.predict_proba(df)
# prediction = prediction_proba.argmax(axis=1)
# prediction_proba = prediction_proba.max()

# st.subheader('Prediction Probability')
# st.write(prediction_proba)
