'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist_test
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from matplotlib import pyplot as plt
from matplotlib import cm

sl = 5
def occlusion_rev(model,n,X_test, Y_test,square_length = sl):
    x = X_test[n].copy()

    ch, s_row, s_col = X_test[n].shape

    heat_array = np.zeros((s_row, s_col))
    pad = square_length // 2 + 1
    occur = np.zeros([28, 28])

    # generate occluded images
    for i in range(s_row):
        # batch s_col occluded images for faster prediction
        for j in range(s_col):


            x_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), 'constant')
            x_pad[:, i:i + square_length, j:j + square_length] = 0.
            x_occluded = x_pad[:, pad:-pad, pad:-pad]
            x_occluded = x - x_occluded

            prob = model.predict_proba(x_occluded[None,...], batch_size=10, verbose=1)

            max_idx = np.argmax(Y_test[n])
            heat_array[i,j] += prob[0,max_idx]
            print(prob)
            occur[i,j] += np.argmax(prob)


    return heat_array, occur


def occlusion(model,n,X_test,Y_test,square_length=sl):

    x = X_test[n].copy()

    ch, s_row, s_col = X_test[n].shape

    heat_array = np.zeros((s_row, s_col))
    pad = square_length // 2 + 1

    # generate occluded images
    for i in range(s_row):
        # batch s_col occluded images for faster prediction
        for j in range(s_col):
            x_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), 'constant')
            x_pad[:, i:i + square_length, j:j + square_length] = 0.
            x_occluded = x_pad[:, pad:-pad, pad:-pad]

            prob = model.predict_proba(x_occluded[None,...], batch_size=10, verbose=1)

            max_idx = np.argmax(Y_test[n])
            heat_array[i,j] += prob[0,max_idx]

    return heat_array


batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist_test.load_data()
print(mnist_test.load_data())

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])




"""
    fig1 = plt.figure()
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.colorbar()
    fig1.savefig(str(i)+'_ori.png')


    heat_array = occlusion(model, i, X_test, Y_test, 5)
    fig = plt.figure()
    plt.imshow(heat_array,cmap='gist_heat')
    plt.colorbar()
    fig.savefig(str(i)+'_heat.png')


"""
simil_map = np.zeros([10,10])
for i in range(0,50):
    fig1 = plt.figure()
    plt.imshow(X_test[i].squeeze(), cmap='gray', interpolation='None')
    plt.colorbar()
    fig1.savefig(str(i)+'_ori.png')


    heat_array = occlusion(model, i, X_test, Y_test, 5)
    fig = plt.figure()
    plt.imshow(heat_array,cmap='gist_heat',interpolation='None')
    plt.colorbar()
    fig.savefig(str(i)+'_heat.png')

    rev_heat_array, occur = occlusion_rev(model,i,X_test,Y_test,5)
    fig = plt.figure()
    plt.imshow(rev_heat_array,cmap='gist_heat', interpolation='None')
    plt.colorbar()
    fig.savefig(str(i)+'_heat_rev.png')
    plt.close('all')

    mask_pos = (rev_heat_array > np.mean(rev_heat_array)).astype(int)
    mask_neg = (rev_heat_array < np.mean(rev_heat_array)).astype(int)
    mask_neg *= -1
    mask_neg -= 1

    mask = mask_pos + mask_neg

    occur = occur * mask
    occur = occur[occur > 0]

    arr = np.zeros(10)
    for j in range(0, occur.size):
        arr[int(occur[j])] += 1

    simil_map[np.argsort(arr)[-1], np.argsort(arr)[-2]] += 1



    #plt.close('all')
def gen_simil_mat(simil_map):

    temp_map = simil_map + simil_map.T

    tri_mask = np.tri(simil_map.shape[0], k=-1)

    simil_tri = np.ma.array(temp_map, mask = tri_mask)
    fig = plt.figure()

    plt.imshow(simil_tri, interpolation=None, cmap='RdPu')
    plt.colorbar()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 10) # jet doesn't have white color
    cmap.set_bad('w') # default value is 'k'
    ax1.imshow(simil_tri, interpolation='None', cmap='RdPu')

    x_ticks = np.arange(0,10,1)
    y_ticks = np.arange(0,10,1)

    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)

    ax1.grid(True)
    ax1.xaxis.tick_top()
    ax1.yaxis.tick_right()
    plt.show()


ori_simil_map = np.zeros([10,10])

for i in range(10000):
    x = X_test[i].copy()
    prob = model.predict_proba(x[None,...],batch_size=10, verbose=1)
    ori_simil_map[np.argsort(prob)[-1][-1], np.argsort(prob)[-1][-2]] += 1

gen_simil_mat(ori_simil_map)