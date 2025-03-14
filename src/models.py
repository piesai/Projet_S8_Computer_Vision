from data_clean import coucou
import keras
import numpy as np
model = keras.Sequential()
model.add(keras.Input(shape=(1,)))
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(1))

train = np.array([1 for k in range(100)] + [0 for k in range(100)])
label = np.array([1 for k in range(100)] + [0 for k in range(100)])
model.compile(optimizer = 'sgd', loss = 'mse')

model.fit(train,label,batch_size=2,epochs=10)

a = model.predict(np.array([[0],[1]]))
print(a)
print(coucou)