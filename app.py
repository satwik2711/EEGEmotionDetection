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

# Load Data
@st.cache
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

# Train Neural Network
def train_neural_network(X_train, y_train):
    model = models.Sequential()
    model.add(layers.Dense(2000, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(3, activation="softmax"))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, batch_size=32)
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

    # SVM Model
    st.subheader("SVM Model")
    svm_model = train_svm(X_train, y_train)
    y_test_labels = y_test.idxmax(axis=1)
    svm_accuracy = svm_model.score(X_test, y_test_labels)
    st.write('SVM Accuracy:', svm_accuracy)
    svm_predictions = svm_model.predict(X_test)
    y_test_multiclass = y_test.idxmax(axis=1)
    plot_confusion_matrix(y_test_multiclass, svm_predictions)

    # Neural Network
    st.subheader("Neural Network")
    nn_model = train_neural_network(X_train, y_train)
    nn_loss, nn_accuracy = nn_model.evaluate(X_test, y_test)
    st.write('Neural Network Accuracy:', nn_accuracy)

    # Note: Add CNN model training and evaluation similarly if needed

if __name__ == "__main__":
    main()
