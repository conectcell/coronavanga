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
optimizer = all.max() / (all.max() / all.mean())
print(optimizer)
all /= optimizer


print(all)


X = all[1:]
y = all[:-1]




from numpy import array
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import Bidirectional








X = array(X).reshape(len(X), 1, 1)



model = Sequential()
model.add(Bidirectional(LSTM(200, activation='relu'), input_shape=(1, 1)))
model.add(Dense(1))
model.compile(optimizer='Nadam', loss='mse', metrics = ['accuracy'])
print(model.summary())
model.fit(X, y, epochs=500, validation_split=0.2, verbose=1, batch_size=8)
#scores = model.evaluate(X_test, y_test)


test_input = array([all[-1]])
test_input = test_input.reshape((1, 1, 1))
test_output = model.predict(test_input, verbose=0)


res = test_output * optimizer
print(res)

