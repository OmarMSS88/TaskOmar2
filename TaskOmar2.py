import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch dataset
glass_identification = fetch_ucirepo(id=42)

# Extract features and targets
X = glass_identification.data.features
y = glass_identification.data.targets

# EDA Graph
st.title("EDA Graph")

# Bar graph showing the frequency of glass types
eda_fig, ax = plt.subplots()
sns.countplot(x=y, ax=ax)
ax.set_title("Frequency of Glass Types")
ax.set_xlabel("Glass Type")
ax.set_ylabel("Frequency")
st.pyplot(eda_fig)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
st.button("Show Random Forest Results")
if st.button:
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train.values.ravel())
    y_pred_rf = rf_classifier.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

    st.subheader("Random Forest Results")
    st.write(f"Accuracy of Random Forest: {accuracy_rf}")
    st.write("Confusion Matrix:")
    st.write(conf_matrix_rf)
    st.pyplot(sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues').figure)

# Support Vector Machine (SVM)
st.button("Show SVM Results")
if st.button:
    svm_classifier = SVC(random_state=42)
    svm_classifier.fit(X_train, y_train.values.ravel())
    y_pred_svm = svm_classifier.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

    st.subheader("SVM Results")
    st.write(f"Accuracy of SVM: {accuracy_svm}")
    st.write("Confusion Matrix:")
    st.write(conf_matrix_svm)
    st.pyplot(sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues').figure)

# K-Nearest Neighbors (KNN)
st.button("Show KNN Results")
if st.button:
    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(X_train, y_train.values.ravel())
    y_pred_knn = knn_classifier.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

    st.subheader("KNN Results")
    st.write(f"Accuracy of KNN: {accuracy_knn}")
    st.write("Confusion Matrix:")
    st.write(conf_matrix_knn)
    st.pyplot(sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues').figure)
