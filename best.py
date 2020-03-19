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
y = all[:-1]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X = np.array(X).reshape(len(X), 1, 1)
X_train = np.array(X_train).reshape(len(X_train), 1, 1)
X_test = np.array(X_test).reshape(len(X_test), 1, 1)

from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense,Bidirectional, LSTM
from keras.wrappers.scikit_learn import KerasClassifier

def create_model(optimizer, activation, type):
    model = Sequential()
    model.add(Bidirectional(LSTM(type, activation=activation), input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer = optimizer, loss='mse', metrics=['accuracy'])
    return model



# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


model = KerasClassifier(build_fn=create_model, verbose=0)
batch_size = [1,2, 5, 10, 20]
epochs = [10, 50, 100, 1000, 2000, 2500 ]


param_grid = dict(batch_size=batch_size, epochs=epochs)


parameters = {'batch_size': [x+1 for x in range(20)],
              'epochs': [5, 10, 20, 50, 100, 500, 1000, 2000 ],
              'type': [50, 100, 200 ],
              'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
              'activation': ['relu','sigmoid','tanh','elu','selu']
              }


grid = GridSearchCV(estimator=model, param_grid=parameters, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y, verbose = 1, validation_split = 0.2)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
