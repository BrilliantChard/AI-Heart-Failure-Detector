import streamlit as st

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

st.title('Data Analysis Using Seaborn')

st.subheader('Description')

st.text('This app uses seaborn for data analysis. Enjoy the moment here at Coutiato')

flight_data = pd.read_csv('flight_data.csv', index_col = 'Month')

st.write('📊 Sample Data')
st.write(flight_data.head())

st.subheader("✈️ Average Arrival Delay for Spirit Airlines Flights, by Month")

fig, ax = plt.subplots()
sns.barplot(x=flight_data.index, y=flight_data['NK'], ax=ax)
ax.set_title('Average Arrival Delay for Spirit Airlines Flights, by Month')
ax.set_ylabel('Arrival delay (in minutes)')
st.pyplot(fig)



