'exec(%matplotlib inline)'
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np



mypath = './download/COVID-19-master/csse_covid_19_data/csse_covid_19_daily_reports'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

li = []

for filename in onlyfiles:
    if '.csv' in filename:
        df = pd.read_csv(mypath+"/"+filename)
        df['Date'] = filename.replace('.csv', '')
        li.append(df)

df = pd.concat(li, axis=0, ignore_index=True,sort=False)


df.drop(['Province/State','Last Update','Latitude','Longitude'], 1, inplace=True)
df.replace('Mainland China', 'China', inplace=True)
df = df.replace(np.nan, 0, regex=True)
df['Date'] = pd.to_datetime(df['Date'])
df['Confirmed'] = df['Confirmed'].astype(int)
df['Recovered'] = df['Confirmed'].astype(int)
df['Deaths'] = df['Deaths'].astype(int)





work = df.groupby('Date', as_index=False)['Confirmed'].sum()
all = np.array(work['Confirmed'], dtype=float)
all /= 200000



X = all[1:]
Y = all[:-1]




from numpy import array
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import Bidirectional


print(X)
print(Y)


X = array(X).reshape(len(X), 1, 1)


model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
print(model.summary())



model.fit(X, Y, epochs=2000, validation_split=0.2, verbose=1, batch_size=5)


test_input = array([all[-1]])
test_input = test_input.reshape((1, 1, 1))
test_output = model.predict(test_input, verbose=0)


res = round((test_output*200000)[0][0]).astype(int)
print("Tomorrow infected number: " + res.astype(str))
