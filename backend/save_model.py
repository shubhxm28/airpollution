import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Training data
data = {
    'Temperature (C)': [22.5, 25.1, 30.3, 15.2, 28.7, 19.0],
    'Humidity (%)': [45, 60, 70, 50, 55, 65],
    'PM2.5 (ug/m³)': [35, 50, 70, 30, 90, 40]
}
df = pd.DataFrame(data)
df['AQI'] = 25.142 + 0.9 * df['Temperature (C)'] + 0.48 * df['Humidity (%)'] + 0.03 * df['PM2.5 (ug/m³)']

X = df[['Temperature (C)', 'Humidity (%)', 'PM2.5 (ug/m³)']]
y = df['AQI']

# Train and save model
model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'linear_model.pkl')
