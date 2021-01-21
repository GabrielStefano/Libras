from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
features_length = 4096
classes = 11

model = Sequential()
model.add(LSTM(1024, return_sequences=False,
               input_shape=(18, features_length),
               dropout=0.5))
# model.add(LSTM(2048, dropout=0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(classes, activation='softmax'))

print(model.summary())