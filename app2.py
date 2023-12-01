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
    st.title("EEG Emotion Detection")

    data = load_data()
    st.write(data.head())

    X_train, X_test, y_train, y_test = preprocess_data(data)
    y_test_labels = y_test.idxmax(axis=1)  # We will use this for all classifiers

    # SVM Model
    st.subheader("SVM Model")
    svm_model = train_svm(X_train, y_train)
    svm_accuracy = svm_model.score(X_test, y_test_labels)
    st.write('SVM Accuracy:', svm_accuracy)
    svm_predictions = svm_model.predict(X_test)
    st.write("SVM Confusion Matrix:")
    plot_confusion_matrix(y_test_labels, svm_predictions)

    # KNN Model
    st.subheader("KNN Model")
    knn_model = train_knn(X_train, y_train)
    knn_accuracy = knn_model.score(X_test, y_test_labels)
    st.write('KNN Accuracy:', knn_accuracy)
    knn_predictions = knn_model.predict(X_test)
    st.write("KNN Confusion Matrix:")
    plot_confusion_matrix(y_test_labels, knn_predictions)

    # LSTM Model
    st.subheader("LSTM Neural Network")
    lstm_model = train_lstm(X_train, y_train, (1, X_train.shape[1]))
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test_reshaped, y_test)
    st.write('LSTM Neural Network Accuracy:', lstm_accuracy)
    lstm_predictions = lstm_model.predict(X_test_reshaped).argmax(axis=1)
    st.write("LSTM Confusion Matrix:")
    plot_confusion_matrix(y_test.idxmax(axis=1), lstm_predictions)

    # Additional Graphs
    st.subheader("Data Distribution")
    label_counts = data['label'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
