import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('./xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Columns from x_train
x_train_columns = [
    'Age', 'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy',
    'Thyroid Function_Clinical Hyperthyroidism',
    'Thyroid Function_Clinical Hypothyroidism',
    'Thyroid Function_Euthyroid',
    'Thyroid Function_Subclinical Hyperthyroidism',
    'Thyroid Function_Subclinical Hypothyroidism',
    'Physical Examination_Diffuse goiter',
    'Physical Examination_Multinodular goiter',
    'Physical Examination_Normal',
    'Physical Examination_Single nodular goiter-left',
    'Physical Examination_Single nodular goiter-right',
    'Adenopathy_Bilateral', 'Adenopathy_Extensive', 'Adenopathy_Left',
    'Adenopathy_No', 'Adenopathy_Posterior', 'Adenopathy_Right',
    'Pathology_Follicular', 'Pathology_Hurthel cell',
    'Pathology_Micropapillary', 'Pathology_Papillary',
    'Focality_Multi-Focal', 'Focality_Uni-Focal', 'Risk_High',
    'Risk_Intermediate', 'Risk_Low', 'T_T1a', 'T_T1b', 'T_T2', 'T_T3a',
    'T_T3b', 'T_T4a', 'T_T4b', 'N_N0', 'N_N1a', 'N_N1b', 'M_M0', 'M_M1',
    'Stage_I', 'Stage_II', 'Stage_III', 'Stage_IVA', 'Stage_IVB',
    'Response_Biochemical Incomplete', 'Response_Excellent',
    'Response_Indeterminate', 'Response_Structural Incomplete'
]

# Title of the app
st.title('Thyroid Cancer Recurrent Prediction Machine Learning App Designed by Dr Joy Aifuobhokhan')

# Sidebar with comments
st.sidebar.title("Welcome")
st.sidebar.write(
    """
    Welcome to the Thyroid Cancer Recurrent Prediction Machine Learning App. 
    This web application serves as a clinical decision support tool, enabling healthcare professionals to predict the recurrence of thyroid cancer 
    based on patient data in real time.
    """
)


def user_input_features():
    # User input collection
    age = st.slider('Age', 0, 120, 30)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    smoking = st.selectbox('Smoking', ['Yes', 'No'])
    hx_smoking = st.selectbox('Hx Smoking', ['Yes', 'No'])
    hx_radiothreapy = st.selectbox('Hx Radiothreapy', ['Yes', 'No'])
    
    # Dynamically collect multi-choice inputs
    thyroid_function = st.selectbox(
        'Thyroid Function',
        ['Clinical Hyperthyroidism', 'Clinical Hypothyroidism', 'Euthyroid', 
         'Subclinical Hyperthyroidism', 'Subclinical Hypothyroidism']
    )
    physical_exam = st.selectbox(
        'Physical Examination',
        ['Diffuse goiter', 'Multinodular goiter', 'Normal', 
         'Single nodular goiter-left', 'Single nodular goiter-right']
    )
    adenopathy = st.selectbox(
        'Adenopathy',
        ['Bilateral', 'Extensive', 'Left', 'No', 'Posterior', 'Right']
    )
    pathology = st.selectbox(
        'Pathology',
        ['Follicular', 'Hurthel cell', 'Micropapillary', 'Papillary']
    )
    focality = st.selectbox('Focality', ['Multi-Focal', 'Uni-Focal'])
    risk = st.selectbox('Risk', ['High', 'Intermediate', 'Low'])
    t = st.selectbox('T', ['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'])
    n = st.selectbox('N', ['N0', 'N1a', 'N1b'])
    m = st.selectbox('M', ['M0', 'M1'])
    stage = st.selectbox('Stage', ['I', 'II', 'III', 'IVA', 'IVB'])
    response = st.selectbox(
        'Response',
        ['Biochemical Incomplete', 'Excellent', 'Indeterminate', 'Structural Incomplete']
    )

    # Encode inputs
    features = {
        'Age': age,
        'Gender': 1 if gender == 'Female' else 0,
        'Smoking': 1 if smoking == 'Yes' else 0,
        'Hx Smoking': 1 if hx_smoking == 'Yes' else 0,
        'Hx Radiothreapy': 1 if hx_radiothreapy == 'Yes' else 0,
    }

    # Dynamically encode multi-choice features
    for col in x_train_columns:
        if col.startswith('Thyroid Function_'):
            features[col] = 1 if col.split('_')[-1] in thyroid_function else 0
        elif col.startswith('Physical Examination_'):
            features[col] = 1 if col.split('_')[-1] in physical_exam else 0
        elif col.startswith('Adenopathy_'):
            features[col] = 1 if col.split('_')[-1] in adenopathy else 0
        elif col.startswith('Pathology_'):
            features[col] = 1 if col.split('_')[-1] in pathology else 0
        elif col.startswith('Focality_'):
            features[col] = 1 if col.split('_')[-1] in focality else 0
        elif col.startswith('Risk_'):
            features[col] = 1 if col.split('_')[-1] in risk else 0
        elif col.startswith('T_'):
            features[col] = 1 if col.split('_')[-1] in t else 0
        elif col.startswith('N_'):
            features[col] = 1 if col.split('_')[-1] in n else 0
        elif col.startswith('M_'):
            features[col] = 1 if col.split('_')[-1] in m else 0
        elif col.startswith('Stage_'):
            features[col] = 1 if col.split('_')[-1] in stage else 0
        elif col.startswith('Response_'):
            features[col] = 1 if col.split('_')[-1] in response else 0

    return pd.DataFrame([features])

# Collect user input features
input_features_df = user_input_features()

# Ensure all columns in x_train are present in the input DataFrame
for col in x_train_columns:
    if col not in input_features_df.columns:
        input_features_df[col] = 0

# Align columns with x_train
input_features_df = input_features_df[x_train_columns]

# Make predictions
prediction = model.predict_proba(input_features_df)[:, 1]

# Display prediction
st.write(f"The predicted probability of recurrence is: {prediction[0]:.2f}")

# Add a submit button to run the model
if st.button('Predict'):
    st.write(f"The model has made a prediction. The predicted recurrence probability is: {prediction[0]:.2f}")