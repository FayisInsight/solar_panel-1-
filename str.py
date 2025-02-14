import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load('LGBM.pkl')

# Load your dataset
df = pd.read_csv("solarpowergeneration.csv")

# Streamlit app
st.title('Solar Power Generation Predictor')

st.sidebar.header('Input Parameters')

def user_input_features():
    # Create input fields for all features used in the model
    temperature = st.sidebar.slider('Temperature (°C)', 
                                  float(df['temperature'].min()), 
                                  float(df['temperature'].max()), 
                                  float(df['temperature'].mean()))
    
    humidity = st.sidebar.slider('Humidity (%)', 
                               float(df['humidity'].min()), 
                               float(df['humidity'].max()), 
                               float(df['humidity'].mean()))
    
    sky_cover = st.sidebar.slider('Sky Cover', 
                                float(df['sky-cover'].min()), 
                                float(df['sky-cover'].max()), 
                                float(df['sky-cover'].mean()))
    
    wind_speed = st.sidebar.slider('Wind Speed (m/s)', 
                                 float(df['wind-speed'].min()), 
                                 float(df['wind-speed'].max()), 
                                 float(df['wind-speed'].mean()))
    
    wind_direction = st.sidebar.slider('Wind Direction (degrees)', 
                                    0.0, 360.0, 180.0)
    
    avg_pressure = st.sidebar.slider('Average Pressure (hPa)', 
                                   float(df['average-pressure-(period)'].min()), 
                                   float(df['average-pressure-(period)'].max()), 
                                   float(df['average-pressure-(period)'].mean()))
    
    solar_noon = st.sidebar.slider('Distance to Solar Noon', 
                                 float(df['distance-to-solar-noon'].min()), 
                                 float(df['distance-to-solar-noon'].max()), 
                                 float(df['distance-to-solar-noon'].mean()))
    
    visibility = st.sidebar.selectbox('Visibility Category', 
                                    ['Low', 'Medium', 'High'], index=1)
    
    # Convert visibility to encoded value
    visibility_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    visibility_encoded = visibility_mapping[visibility]
    
    # Convert wind direction to radians
    wind_direction_rad = np.deg2rad(wind_direction)
    
    data = {
        'temperature': temperature,
        'humidity': humidity,
        'sky-cover': sky_cover,
        'wind-speed': wind_speed,
        'wind-direction': wind_direction_rad,
        'average-pressure-(period)': avg_pressure,
        'distance-to-solar-noon': solar_noon,
        'visibility_encoded': visibility_encoded
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display input parameters
st.subheader('Selected Input Parameters')
st.write(input_df)

# Prediction
prediction = model.predict(input_df)

st.subheader('Prediction')
st.markdown(f"**Predicted Power Generated:** {prediction[0]:.2f} MW")

# EDA Visualizations
st.header('Exploratory Data Analysis')

# Correlation Heatmap
if st.checkbox('Show Correlation Heatmap'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)

# Pairplot
if st.checkbox('Show Pairplot'):
    st.write(sns.pairplot(df))
    st.pyplot()

# Boxplots
if st.checkbox('Show Boxplots'):
    num_cols = len(df.columns)
    fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(12, 5 * num_cols))
    
    for ax, column in zip(axes, df.columns):
        sns.boxplot(x=df[column], ax=ax, color='skyblue')
        ax.set_title(f'Boxplot of {column}', fontsize=12)
    
    st.pyplot(plt)

# Model Performance
st.header('Model Performance')

performance_data = {
    'Model': ['XGB Regressor', 'Gradient Boosting', 'LightGBM'],
    'R² Score': [0.86, 0.85, 0.88],  # Replace with your actual values
    'MSE': [0.45, 0.48, 0.42]       # Replace with your actual values
}

st.table(pd.DataFrame(performance_data))

# Feature Importance
st.subheader('Feature Importance')
importance_df = pd.DataFrame({
    'Feature': ['distance-to-solar-noon', 'humidity', 'temperature', 
               'wind-speed', 'sky-cover', 'average-pressure-(period)',
               'wind-direction', 'visibility_encoded'],
    'Importance': [0.25, 0.20, 0.18, 0.15, 0.10, 0.07, 0.04, 0.01]
})  # Replace with your actual importance values

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
st.pyplot(plt)
