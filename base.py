import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


#IMPORTANDO DATASET
df = pd.read_csv("BTC_USDT_ohlcv.csv")

#LIMPIANDO DATOS
df = df.dropna()  # Drop rows with missing values
df = df.reset_index(drop=True)  # Reset index after dropping rows

#SEPARANDO DATOS PARA ENTRENAMIENTO Y PRUEBA
dataframe_training, dataframe_testing, y_train, y_test = train_test_split(
    df, df, test_size=0.2, random_state=42
)


#preprocesando los datos, y escalandolos.
##OBTIENE COLUMNA DE PRECIO DE APERTURA
training_set = dataframe_training.iloc[:,1:2].values
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

testing_set = dataframe_testing.iloc[:,1:2].values

#configurar el timestep y salto de tiempo
timesteps = 100
salto = 1


#preparando las series temporales
print("preparando las series temporales")
X_train = []
Y_train = []
print(len(training_set_scaled))
for i in range(timesteps, len(training_set_scaled)):
    #APPEND INTO X_TRAIN: ROWS, COLUMN
    X_train.append(training_set_scaled[i-timesteps:i, 0]) #MEMORY
    Y_train.append(training_set_scaled[i+salto-1, 0]) #PREDICTIONS
    print(i)


print(X_train)
print(Y_train)

