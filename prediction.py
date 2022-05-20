import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import joblib
import streamlit as st


# Page title and layout
st.set_page_config(page_title='Iris Flower Classification', layout='wide')
st.header("Iris Flower Classification")

# Loading the dataset
df = pd.read_csv('Iris.csv')
df.drop(columns=['Id'], inplace=True)
data = df.drop(columns=['Species'])
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, stratify=y, random_state=99)
scalar = StandardScaler()
X_train_scale = scalar.fit_transform(X_train)
# X_test_scale = scalar.transform(X_test)

model = joblib.load('LogisticRegression.pkl')

# Prediction function
def predict(values):
    values_scaled = scalar.transform(values)
    result = model.predict(values_scaled)[0] #string
    return result

# Input    
if st.checkbox('Show dataframe'):
    df   
    
SepalLengthCm = st.slider('Sepal length(cm)', 0.1, max(df["SepalLengthCm"]))
SepalWidthCm = st.slider('Sepal width(cm)', 0.1, max(df["SepalWidthCm"]))
PetalLengthCm = st.slider('Petal length(cm)', 0.1, max(df["PetalLengthCm"]))
PetalWidthCm = st.slider('Petal width (cm) ', 0.1, max(df["PetalWidthCm"]))

if st.button('Make Prediction'):
    iris_class = predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
    st.write(f"The class of the Iris flower is {iris_class}")
    
