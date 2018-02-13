import numpy as np
import time
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomNormal
from keras import optimizers
from keras import regularizers

from keras.callbacks import EarlyStopping
from keras.callbacks import LambdaCallback

from sklearn.model_selection import StratifiedKFold



three_layers = False
noise_stddev = 0.18


# Sets the parameters for the Mackey-Glass time serie
beta = 0.2
gamma = 0.1
n = 10
tau = 25

number_of_points = 1506

data = np.zeros(number_of_points)
data[0] = 1.5

# Generates the data using Euler's method for the Mackey-Glass time serie
for i in range(number_of_points-1):

    data[i+1] = 0.9*data[i]

    if (i > 24):
        data[i+1] += (0.2*data[i-25])/(1 + pow(data[i-25],n))


# Plots the time serie
X_array = np.linspace(0,number_of_points-1,number_of_points)
plt.title('Mackey-Glass time serie')
plt.xlabel('time')
plt.plot(X_array, data, '--')
#plt.show()

# Adds zero-mean Gaussian noise
data[281:1506] = data[281:1506] + np.random.normal(0, noise_stddev, 1225)



# Constructs data sets
input_sequence = np.array([20,15,10,5,0])

X_whole = np.zeros((5,1200))

X_whole[0,:] = data[301-input_sequence[0]:1501-input_sequence[0]]
X_whole[1,:] = data[301-input_sequence[1]:1501-input_sequence[1]]
X_whole[2,:] = data[301-input_sequence[2]:1501-input_sequence[2]]
X_whole[3,:] = data[301-input_sequence[3]:1501-input_sequence[3]]
X_whole[4,:] = data[301-input_sequence[4]:1501-input_sequence[4]]

X_train = np.zeros((5,1000))
X_test = np.zeros((5,200))

X_train = X_whole[:,:1000]
X_test = X_whole[:,1000:1200]

Y_train = np.zeros(1000)
Y_test = np.zeros(200)

Y_train = data[306:1306]
Y_test = data[1306:1506]



# Keeps track of the elapsed time
start_time = time.time()

# Creates neural network
model = Sequential()

activation_function = 'relu'
first_layer_nodes = 8
second_layer_nodes = 8
third_layer_nodes = 8
reg_factor = 0


model.add(Dense(units=first_layer_nodes, activation=activation_function, use_bias = 'TRUE', \
    kernel_initializer=RandomNormal(mean=0.0, stddev=1./np.sqrt(first_layer_nodes)), \
    bias_initializer=RandomNormal(mean=0.0, stddev=1./np.sqrt(first_layer_nodes)), input_dim=5, \
    kernel_regularizer=regularizers.l2(reg_factor), bias_regularizer=regularizers.l2(reg_factor)))
model.add(Dense(units=second_layer_nodes, activation=activation_function, use_bias = 'TRUE', \
    kernel_initializer=RandomNormal(mean=0.0, stddev=1./np.sqrt(second_layer_nodes)), \
    bias_initializer=RandomNormal(mean=0.0, stddev=1./np.sqrt(second_layer_nodes)), \
    kernel_regularizer=regularizers.l2(reg_factor), bias_regularizer=regularizers.l2(reg_factor)))
if (third_layer_nodes == True):
    model.add(Dense(units=third_layer_nodes, activation=activation_function, use_bias = 'TRUE', \
        kernel_initializer=RandomNormal(mean=0.0, stddev=1./np.sqrt(third_layer_nodes)), \
        bias_initializer=RandomNormal(mean=0.0, stddev=1./np.sqrt(third_layer_nodes)), \
        kernel_regularizer=regularizers.l2(reg_factor), bias_regularizer=regularizers.l2(reg_factor)))
model.add(Dense(units=1, activation='relu', use_bias = 'TRUE', \
    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=1), \
    kernel_regularizer=regularizers.l2(reg_factor), bias_regularizer=regularizers.l2(reg_factor)))


# Learning rate (lr) is the eta parameter for backprop, sgd means stochastic gradient descent
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=sgd)

# patience is the number of epochs without improvement before which we stop
# min_delta is the minimum value of the improvement (on the smallest value already reached) in order to be considered as such
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=10)

# batch_size of 1 for sequential learning and len(X_train) for batch learning
# validation_split takes the last elements of the training set without shuffling them
history = model.fit(np.transpose(X_train), np.transpose(Y_train), validation_split=0.5, \
    epochs=500, batch_size=len(X_train), verbose=0, shuffle = 'FALSE', callbacks = [early_stopping])


# Displays total time of the simulation
elapsed_time = time.time() - start_time
print("The time of the simulation is: " + str(elapsed_time) + " seconds.")

# Rank 2k for the weights of layer k and rank 2k+1 fro the bias of layer k
final_weights = model.get_weights()

# Gets the final validation loss
output = history.history
validation_loss = output["val_loss"]
print("The mean square error on the validation set is: " + str(validation_loss[-1]))

# Gets the test loss
test_score = model.evaluate(np.transpose(X_test), np.transpose(Y_test), verbose=0)
print("The mean square error on the test set is: " + str(test_score))


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training','validation'], loc='upper right')
#plt.show()

Y_pred = model.predict(np.transpose(X_test))
plt.plot(Y_pred)
plt.plot(Y_test)
plt.title('Predictions on test set')
plt.legend(['model prediction','test values'], loc='upper right')
#plt.show()



# Implements K-fold validation using scikit-learn
"""
seed = 7
np.random.seed(seed)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for train, test in kfold.split(X, Y):
    model = Sequential()
    model.add(Dense(64, input_dim=12, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.fit(X[train], Y[train], epochs=10, verbose=1)
"""
