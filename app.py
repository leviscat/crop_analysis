import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load('crop_model.pkl')
df = pd.read_csv('Crop_recommendation.csv')

st.title("Crop Recommendation System")
st.write("Adjust soil and climate parameters to get a crop recommendation.")

N = st.slider("Nitrogen (N)", 0, 140, 50)
P = st.slider("Phosphorus (P)", 5, 145, 50)
K = st.slider("Potassium (K)", 5, 205, 50)
temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0)
humidity = st.slider("Humidity (%)", 14.0, 100.0, 60.0)
ph = st.slider("pH", 3.5, 10.0, 6.5)
rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                           columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

prediction = model.predict(input_data)[0]
st.success(f"Recommended Crop: {prediction}")

st.subheader(f"Your input vs typical {prediction} conditions")
crop_avg = df[df['label'] == prediction][['N','P','K','temperature','humidity','ph','rainfall']].mean()
user_input = pd.Series([N, P, K, temperature, humidity, ph, rainfall],
                        index=['N','P','K','temperature','humidity','ph','rainfall'])

fig, ax = plt.subplots(figsize=(10, 4))
x = range(len(crop_avg))
ax.bar([i - 0.2 for i in x], crop_avg.values, width=0.4, label=f'Typical {prediction}')
ax.bar([i + 0.2 for i in x], user_input.values, width=0.4, label='Your input')
ax.set_xticks(list(x))
ax.set_xticklabels(crop_avg.index)
ax.legend()
st.pyplot(fig)