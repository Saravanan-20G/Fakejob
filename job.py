import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
best_model = joblib.load('C:\\Users\\Saravanan\\OneDrive\\Desktop\\ipl\\random_forest_model.pkl')
vectorizer = joblib.load('C:\\Users\\Saravanan\\OneDrive\\Desktop\\ipl\\vectorizer.pkl')

# Load the dataset for dropdowns
df = pd.read_csv('C:\\Users\\Saravanan\\OneDrive\\Desktop\\ipl\\job_predict.csv')

# Define prediction function
def predict_fraud(title, company_profile, description, requirements, benefits, employment_type, required_experience, required_education, industry, function):
    text = ' '.join([title, company_profile, description, requirements, benefits])
    text_features = vectorizer.transform([text])
    other_features = pd.DataFrame([[employment_type, required_experience, required_education, industry, function]], 
                                  columns=['employment_type', 'required_experience', 'required_education', 'industry', 'function'])
    
    # Convert all column names to strings
    other_features.columns = other_features.columns.astype(str)
    text_features_df = pd.DataFrame(text_features.toarray())
    text_features_df.columns = text_features_df.columns.astype(str)
    
    features = pd.concat([other_features, text_features_df], axis=1)
    prediction = best_model.predict(features)
    return prediction[0]

# Streamlit app
st.title('Job Posting Fraud Detection')
title = st.text_input('Job Title')
company_profile = st.text_area('Company Profile')
description = st.text_area('Job Description')
requirements = st.text_area('Job Requirements')
benefits = st.text_area('Job Benefits')
employment_type = st.selectbox('Employment Type', df['employment_type'].unique())
required_experience = st.selectbox('Required Experience', df['required_experience'].unique())
required_education = st.selectbox('Required Education', df['required_education'].unique())
industry = st.selectbox('Industry', df['industry'].unique())
function = st.selectbox('Function', df['function'].unique())

if st.button('Predict'):
    result = predict_fraud(title, company_profile, description, requirements, benefits, employment_type, required_experience, required_education, industry, function)
    st.write('Prediction:', 'Fake' if result == 1 else 'Real')
