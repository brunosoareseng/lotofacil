import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout

df = pd.read_excel("./lotofacil.xlsx")
df.describe()

df.drop(['Concurso', 'Data'], axis=1, inplace=True)
#df.set_index('Data')
df.head()

# Retira ultimo resultado para treinar
dft = df.copy()
dft.drop(dft.tail(1).index, inplace=True)

scaler = StandardScaler().fit(dft.values)
transformed_dataset = scaler.transform(dft.values)
transformed_dft = pd.DataFrame(data=transformed_dataset, index=dft.index)
transformed_dft.head()

number_of_rows = dft.values.shape[0]
window_length = 14
number_of_features = dft.values.shape[1]

X = np.empty([number_of_rows - window_length, window_length, number_of_features], dtype=float)
y = np.empty([number_of_rows - window_length, number_of_features], dtype=float)
for i in range(0, number_of_rows-window_length):
    X[i] = transformed_dft.iloc[i:i+window_length, 0:number_of_features]
    y[i] = transformed_dft.iloc[i+window_length:i+window_length+1, 0:number_of_features]

model = Sequential()
model.add(Bidirectional(LSTM(240, input_shape = (window_length, number_of_features), return_sequences = True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240, input_shape = (window_length, number_of_features), return_sequences = True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240, input_shape = (window_length, number_of_features), return_sequences = True)))
model.add(Bidirectional(LSTM(240, input_shape = (window_length, number_of_features), return_sequences = False)))
model.add(Dense(180))
model.add(Dense(60))
model.add(Dense(number_of_features))

from tensorflow import keras
from keras.optimizers import Adam

values = []

for i in range(10):

    model.compile(optimizer=Adam(learning_rate=0.0002), loss ='mse', metrics=['accuracy'])

    model.fit(x=X, y=y, batch_size=128, epochs=220, verbose=2)


    ####
    # Evaluate
    ####
    to_predict = df.tail(8)

    to_predict.drop([to_predict.index[-1]],axis=0, inplace=True)

    prediction = df.tail(1)

    to_predict = np.array(to_predict)

    scaled_to_predict = scaler.transform(to_predict)

    y_pred = model.predict(np.array([scaled_to_predict]))
    print("The predicted numbers in the last lottery game are:", scaler.inverse_transform(y_pred).astype(int)[0])

    prediction = np.array(prediction)
    print("The actual numbers in the last lottery game were:", prediction[0])


    ####
    # Predict next
    ####
    to_predict = df.tail(14)

    print(to_predict)

    to_predict = np.array(to_predict)

    scaled_to_predict = scaler.transform(to_predict)
    y_pred = model.predict(np.array([scaled_to_predict]))
    values.append(scaler.inverse_transform(y_pred).astype(int)[0])
    print("The predicted numbers to next lottery game are:", values[i])

    

saida = pd.DataFrame(values)
print(saida)
saida.to_excel("aposta.xlsx")

