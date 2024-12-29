#!/usr/bin/env python
# coding: utf-8

# In[30]:


import sys
sys.path.insert(1, './SafeML/Implementation_in_Python')

from CVM_Distance import CVM_Dist as Cramer_Von_Mises_Dist
from Anderson_Darling_Distance import Anderson_Darling_Dist
from Kolmogorov_Smirnov_Distance import Kolmogorov_Smirnov_Dist
from KuiperDistance import Kuiper_Dist
from WassersteinDistance import Wasserstein_Dist
from DTS_Distance import DTS_Dist 


# In[31]:


import streamlit as st
import logging
from joblib import load


# In[32]:


import os, sys 
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

import pandas as pd 
import seaborn as sns # Graphing, built ontop of MatPlot for ease-of-use and nicer diagrams.
import matplotlib.pyplot as plt # MatPlotLib for graphing data visually. Seaborn more likely to be used.
import numpy as np # For manipulating arrays and changing data into correct formats for certain libraries
import sklearn # For Machine Learning algorithms
import scikitplot # Confusion matrix plotting
from sklearn.decomposition import PCA # For PCA dimensionality reduction technique
from sklearn.preprocessing import StandardScaler # For scaling to unit scale, before PCA application
from sklearn.preprocessing import LabelBinarizer # For converting categorical data into numeric, for modeling stage
from sklearn.model_selection import StratifiedKFold # For optimal train_test splitting, for model input data
from sklearn.model_selection import train_test_split # For basic dataset splitting
from sklearn.neighbors import KNeighborsClassifier # K-Nearest Neighbors ML classifier (default n. of neighbors = 5)
from scikitplot.metrics import plot_confusion_matrix # For plotting confusion matrices
from sklearn.metrics import accuracy_score # For getting the accuracy of a model's predictions
from sklearn.metrics import classification_report # Various metrics for model performance
from sklearn.neural_network import MLPClassifier # For Neural Network classifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


# In[33]:


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep]


# In[34]:


def get_PCA_feature_names(num_of_pca_components):
    feature_names = []
    for i in range(num_of_pca_components):    
        feature_names.append(f"Principal component {i+1}")
    return feature_names


# In[35]:


# 'Reduced dimensions' variable for altering the number of PCA principal components. Can be altered for needs.
# Only 7 principal components needed when using non-normalised PCA dataset.
dimensions_num_for_PCA = 7


# In[40]:


def data_processor (df):
     #Fixing column name issues
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    df.head()
    
    df.dtypes
    
    df_cleaned = df.copy()
    df_cleaned = clean_dataset(df_cleaned) # see methods at top of notebook
    df_cleaned = df_cleaned.reset_index()
    # Removing un-needed index column added by reset_index method
    df_cleaned.drop('index', axis=1, inplace=True)
    
    original_columns = df.columns.tolist()
    
    # Saving the label attribute before dropping it.
    df_labels = df_cleaned['label']
    # Shows all the possible labels/ classes a model can predict.
    # Need to alter these to numeric 0, 1, etc... for model comprehension (e.g. pd.get_dummies()).
    df_labels.unique()
    
    # Axis=1 means columns. Axis=0 means rows. inplace=False means that the original 'df' isn't altered.
    df_no_labels = df_cleaned.drop('label', axis=1, inplace=False)
    # Getting feature names for the StandardScaler process
    df_features = df_no_labels.columns.tolist()
    # Printing out Dataframe with no label column, to show successful dropping
    
    df_scaled = StandardScaler().fit_transform(df_no_labels)
    # Converting back to dataframe
    df_scaled = pd.DataFrame(data = df_scaled, columns = df_features)
    
    pca_test = PCA().fit(df_scaled)
    plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()
    
    # The df_no_labels dataset holds the un-normalised dataset.
    pca_test = PCA().fit(df_no_labels)
    plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()
    
    pca = PCA(n_components=dimensions_num_for_PCA)
    #principal_components = pca.fit(df_scaled).transform(df_scaled) => for normalised PCA

    # Non-normalised PCA
    principal_components = pca.fit(df_no_labels).transform(df_no_labels)
    principal_components
    
    principal_component_headings = get_PCA_feature_names(dimensions_num_for_PCA)
    pca_columns = [f"{col}" for col in original_columns[:dimensions_num_for_PCA]]
    df_pc = pd.DataFrame(data = principal_components, columns = pca_columns)
    df_final = pd.concat([df_pc, df_labels], axis = 1)
    
    
    lb = LabelBinarizer()
    df_final['label'] = lb.fit_transform(df_final['label'])
    df_final

    return df_final


# In[41]:


# Function to perform predictions and log alerts
def predict_and_log(model, X_simulated, threshold=0.8):
    try:
        predictions = model.predict(X_simulated)
        probabilities = model.predict_proba(X_simulated)

        if probabilities.shape[1] <= 1:
            logging.error("Probability matrix has less than 2 columns")
            return

        alert_messages = []
        for i, (prediction, prob) in enumerate(zip(predictions, probabilities[:, 1])):
            if prediction == 1 and prob > threshold:
                alert_message = f"Alert: Suspicious activity detected in sample {i} with probability {prob:.2f}"
                alert_messages.append(alert_message)
                logging.info(alert_message)

        # Displaying results on Streamlit
        if alert_messages:
            for message in alert_messages:
                st.error(message)
        else:
            st.success("No suspicious activity detected.")

    except Exception as e:
        logging.error(f"An error occurred during prediction or logging: {str(e)}")
        st.error(f"An error occurred: {str(e)}")


# In[42]:


# Optional: Add a slider for adjusting the threshold dynamically
threshold = st.slider('Select the probability threshold for alerts', 0.0, 1.0, 0.8)


# In[43]:


#`model` is already imported and trained
model = load('knn_model.pkl')

# Configure logging to ensure it is setup correctly
if not logging.getLogger().hasHandlers():
    logging.basicConfig(filename='alerts.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Streamlit UI setup
st.title("Real-time Prediction Monitor")
st.write("This application performs real-time predictions and logs alerts for suspicious activities.")

# If you have predefined data, you can use that, or let the user input data
# For demonstration, let's assume user inputs data manually or uploads a CSV file
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_final = data_processor (df)
    X_simulated = df_final.drop('label', axis=1)
    if st.button("Predict"):
        predict_and_log(model, X_simulated)


# In[ ]:




