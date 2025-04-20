import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('lb_gender.pkl','rb') as file:
    lb_gender = pickle.load(file)

with open('Ohe_geo.pkl','rb') as file:
    Ohe_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


## Streamlit app
st.title('Customer Churn Prediction')

# User Input
Credit_Score= st.number_input('Credit Score')
Gender=  st.selectbox('Gender', lb_gender.classes_)
Age= st.slider('Age',18,100)
Tenure= st.slider('Tenure',0,10)
Balance= st.number_input('Balance')
Num_of_Products= st.slider('Number of Products',1,4)
Has_Cr_Card= st.selectbox('Has Credit Card',[0,1])
Is_Active_Member= st.selectbox('Is Active Member',[0,1])
Estimated_Salary= st.number_input('Estimated Salary')
Geography=  st.selectbox('Geography', Ohe_geo.categories_[0])

# Prepare the input data
input_data = pd.DataFrame({
	'CreditScore': [Credit_Score],
	'Gender':[lb_gender.transform([Gender])[0]],
	'Age':	[Age],
	'Tenure':[Tenure]	,
	'Balance':[Balance],
	'NumOfProducts':[Num_of_Products],
	'HasCrCard':[Has_Cr_Card],
	'IsActiveMember':[Is_Active_Member],
	'EstimatedSalary':[Estimated_Salary]
})

# One hot Encode 'Geography'
geo_encoded = Ohe_geo.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=Ohe_geo.get_feature_names_out(['Geography']))


# Combining one hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict Churn
prediction = model.predict(input_data_scaled)
prediction_proba =  prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
