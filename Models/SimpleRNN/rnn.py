from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from process import split_window_data


dtframe = pd.read_csv('./ProcessedData/Ban Don.csv')
dtframe= dtframe.drop('Time',axis=1)
cols = ['H','Q']

split_time = 90000
window_size = 6

time = range(dtframe.shape[0])

x_train = dtframe.iloc[:split_time,:]
#print(type(x_train))
x_valid = dtframe.iloc[split_time:,:]
time_valid = time[split_time:]

dataset = split_window_data(dtframe = x_train,cols = cols)
print(list(dataset.__iter__())[:1])

model = keras.models.Sequential([
  keras.layers.SimpleRNN(40,input_dim=[None], return_sequences=True),
  keras.layers.SimpleRNN(40),
  keras.layers.Dense(2)
])

optimizer = keras.optimizers.SGD(lr=5e-6, momentum=0.9)
model.compile(loss= keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(dataset,epochs=400)

forecast=[]
for time in range(dtframe.shape[0] - window_size):
    forecast.append(model.predict(dtframe.iloc[time:time + window_size]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]


plt.figure(figsize=(10, 6))
plt.plot(x_valid['H'],x_valid['Q'])
plt.plot(r