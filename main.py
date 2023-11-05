import keras
import numpy as np

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]) # units is the number of neurons, input_shape is the shape of the input data so in this case just one (xs)

model.compile(optimizer='sgd', loss='mean_squared_error') 

xs = np.array([1, 2, 3, 4, 5, 6], dtype=float) 
ys = np.array([1, 1.5, 2, 2.5, 3, 3.5], dtype=float)

model.fit(xs, ys, epochs=500) # literally saying fit the xs into the ys 500 times, epochs is the number of times the model will see the same data

print(model.predict([10.0])) # prints the prediction for the value 10.0, not exactly correct due to only having 6 values to train on
