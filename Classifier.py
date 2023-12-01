from collections import Counter
from sklearn.inspection import permutation_importance
import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from tensorflow.keras import models, layers
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.layers import LSTM, Dropout
import streamlit as st
from streamlit_lottie import st_lottie
from sklearn.metrics import classification_report

import requests
import io

# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.set_page_config(page_title='EEG Emotion Detection', page_icon=':brain:', layout='wide')

lottie_animation = load_lottieurl("https://lottie.host/dbffacee-8ec8-4ea8-8143-a03c0f5823cb/MtvjsARASY.json")
st_lottie(lottie_animation, height=300, key="emotion")


st.markdown("""
    <style>
    .css-18e3th9 {
        padding-top: 0px;
    }
    .css-1d391kg {
        padding-top: 0px;
    }
    </style>
    """, unsafe_allow_html=True)


# Load Data
@st.cache_data
def load_data():
    # Update the path to where your data is located
    emotion_data = pd.read_csv("emotions.csv")
    return emotion_data

# Data preprocessing
def preprocess_data(data):
    maps = {"NEGATIVE": 2, "POSITIVE": 1, "NEUTRAL": 0}
    y = pd.get_dummies(data['label'].apply(lambda x: maps[x]))
    X = data.drop('label', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

# Train SVM Model
def train_svm(X_train, y_train):
    clf = SVC()
    y_train_labels = y_train.idxmax(axis=1)
    clf.fit(X_train, y_train_labels)
    return clf

# Train KNN Model
def train_knn(X_train, y_train):
    y_train_labels = y_train.idxmax(axis=1)
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train_labels)
    return knn_model

# Train LSTM Model
def train_lstm(X_train, y_train, input_shape):
    # Reshape input data to 3D for LSTM as needed (samples, time steps, features)
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

    model = models.Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_reshaped, y_train, epochs=30, batch_size=32)
    return model



# Display Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    st.pyplot(plt)

# Main function
def main():
    st.title("EEG Emotion Detection 🤔")

    # Sidebar for navigation or global settings (if any)
    st.sidebar.title("Navigation")
    st.sidebar.info("This is a Streamlit web app that compares different machine learning models on EEG emotion detection data.")

    # Data loading and preprocessing
    st.header("Data Overview")
    data = load_data()
    st.dataframe(data.head())

    X_train, X_test, y_train, y_test = preprocess_data(data)
    y_test_labels = y_test.idxmax(axis=1)  # Used for all classifiers

    # Container for model results and confusion matrices
    results_container = st.container()

    with results_container:
        st.header("Model Comparison")

        # Use columns to display accuracy metrics side by side
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("SVM Model")
            svm_model = train_svm(X_train, y_train)
            svm_accuracy = svm_model.score(X_test, y_test_labels)
            st.metric(label="Accuracy", value=f"{svm_accuracy:.2%}")
            svm_predictions = svm_model.predict(X_test)
            st.write("Confusion Matrix:")
            plot_confusion_matrix(y_test_labels, svm_predictions)
            st.text("Classification Report:")
            st.text(classification_report(y_test_labels, svm_predictions))
            
            

        with col2:
            st.subheader("KNN Model")
            knn_model = train_knn(X_train, y_train)
            knn_accuracy = knn_model.score(X_test, y_test_labels)
            st.metric(label="Accuracy", value=f"{knn_accuracy:.2%}")
            knn_predictions = knn_model.predict(X_test)
            st.write("Confusion Matrix:")
            plot_confusion_matrix(y_test_labels, knn_predictions)
            st.text("Classification Report:")
            st.text(classification_report(y_test_labels, knn_predictions))

            
        with col3:
            st.subheader("LSTM")
            lstm_model = train_lstm(X_train, y_train, (1, X_train.shape[1]))
            X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test_reshaped, y_test)
            st.metric(label="Accuracy", value=f"{lstm_accuracy:.2%}")
            lstm_predictions = lstm_model.predict(X_test_reshaped).argmax(axis=1)
            st.write("Confusion Matrix:")
            plot_confusion_matrix(y_test.idxmax(axis=1), lstm_predictions)
            st.text("Classification Report:")
            st.text(classification_report(y_test.idxmax(axis=1), lstm_predictions))

    # Additional Graphs
    st.header("Additional Insights")
    st.subheader("Emotion Distribution")
    label_counts = data['label'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax)
    ax.set_title("Count of Each Emotion in the Dataset")
    st.pyplot(fig)

    st.header("Emotion Classification from EEG Data 🔎")
    uploaded_file = st.file_uploader("Upload your EEG data CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read and preprocess the uploaded file
        uploaded_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(uploaded_data.head())

        # Assuming the uploaded file has the same format as your training data
        # Preprocess the uploaded data
        # _, X_uploaded_scaled, _, _ = preprocess_data(uploaded_data)
        scaler = MinMaxScaler()
        X_uploaded_scaled = scaler.fit_transform(uploaded_data)
        

        # KNN Prediction
        from scipy import stats

# KNN Prediction
        knn_predictions = knn_model.predict(X_uploaded_scaled)

# Find the mode of the predictions
        counter = Counter(knn_predictions)
        most_common_pred = counter.most_common(1)[0][0]

# Define your emotion mapping
        emotion_mapping = {
            2: "Negative 😔",
            1: "Positive 😊",
            0: "Neutral 😐"
        }


        most_common_emotion = emotion_mapping[most_common_pred]

        st.write("The most common KNN predicted emotion:")
        st.title(most_common_emotion)




    # Theme and layout adjustments
   


if __name__ == "__main__":
    main()
