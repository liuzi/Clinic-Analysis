import tensorflow as tf
import keras

from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential, Model
from keras import layers
from keras import optimizers
from keras.layers import  BatchNormalization, Activation,  Dense, Input, Dropout
from sklearn.preprocessing import minmax_scale

# NOTE: Test whether tensorflow can work
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def reduce_dimension(inputdata, new_dimension):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    set_session(tf.Session(config=config))

    ## Build autoencoder model
    model = Sequential()
    model.add(Dense(800,  activation='relu', input_shape=(2138,)))
    model.add(BatchNormalization())
    # model.add(Dense(800, activation='relu'))
    model.add(Dense(128,  activation='relu', name="bottleneck"))
    model.add(BatchNormalization())
    model.add(Dense(800, activation='relu'))
    # model.add(Dense(1600,  activation='relu'))
    model.add(Dense(2138,  activation='sigmoid'))
    model.summary()

    encoder = Model(model.input, model.get_layer('bottleneck').output)
    model.compile(loss='binary_crossentropy', optimizer = optimizers.Adadelta())
    model.fit(inputdata,inputdata, batch_size=128, epochs=500,shuffle=True)

    repre_128 = encoder.predict(inputdata) 
    # repre_128.shape
    write2file(pd.DataFrame(repre_128), join(write_prefix,"Autoencoder_128_v2"))


def preprocess_inputdata(test1, *inputdatum):
    for i in inputdatum:
        print(i)


preprocess_inputdata("aa", "bb", "dd", "ee")


