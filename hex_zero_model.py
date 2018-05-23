from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

cnn_filter_num = 128
cnn_first_filter_size = 2
cnn_filter_size = 2
l2_reg = 0.0001
res_layer_num = 20
n_labels = 64
value_fc_size = 64
learning_rate = 0.1 # schedule dependent on thousands of steps, every 200 thousand steps, decrease by factor of 10
momentum = 0.9


def build_model():
    """
    Builds the full Keras model and returns it.
    """
    in_x = x = Input((1, 8, 8))

    # (batch, channels, height, width)
    x = Conv2D(filters=cnn_filter_num,   kernel_size=cnn_first_filter_size, padding="same", data_format="channels_first", use_bias=False, kernel_regularizer=l2(l2_reg), name="input_conv-"+str(cnn_first_filter_size)+"-"+str(cnn_filter_num))(x)
    x = BatchNormalization(axis=1, name="input_batchnorm")(x)
    x = Activation("relu", name="input_relu")(x)

    for i in range(res_layer_num):
        x = _build_residual_block(x, i + 1)

    res_out = x

    # for policy output
    x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=l2(l2_reg), name="policy_conv-1-2")(res_out)

    x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
    x = Activation("relu", name="policy_relu")(x)
    x = Flatten(name="policy_flatten")(x)

    # no output for 'pass'
    policy_out = Dense(n_labels, kernel_regularizer=l2(l2_reg), activation="softmax", name="policy_out")(x)

    # for value output
    x = Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=l2(l2_reg), name="value_conv-1-4")(res_out)

    x = BatchNormalization(axis=1, name="value_batchnorm")(x)
    x = Activation("relu",name="value_relu")(x)
    x = Flatten(name="value_flatten")(x)
    x = Dense(value_fc_size, kernel_regularizer=l2(l2_reg), activation="relu", name="value_dense")(x)

    value_out = Dense(1, kernel_regularizer=l2(l2_reg), activation="tanh", name="value_out")(x)

    model = Model(in_x, [policy_out, value_out], name="hex_model")

    sgd = optimizers.SGD(lr=learning_rate, momentum=momentum)

    losses = ['categorical_crossentropy', 'mean_squared_error']

    model.compile(loss=losses, optimizer='adam', metrics=['accuracy', 'mae'])

    model.summary()
    return model

def _build_residual_block(x, index):
    in_x = x
    res_name = "res"+str(index)
    x = Conv2D(filters=cnn_filter_num, kernel_size=cnn_filter_size, padding="same", data_format="channels_first", use_bias=False, kernel_regularizer=l2(l2_reg), name=res_name+"_conv1-"+str(cnn_filter_size)+"-"+str(cnn_filter_num))(x)
    x = BatchNormalization(axis=1, name=res_name+"_batchnorm1")(x)
    x = Activation("relu",name=res_name+"_relu1")(x)
    x = Conv2D(filters=cnn_filter_num, kernel_size=cnn_filter_size, padding="same", data_format="channels_first", use_bias=False, kernel_regularizer=l2(l2_reg), name=res_name+"_conv2-"+str(cnn_filter_size)+"-"+str(cnn_filter_num))(x)
    x = BatchNormalization(axis=1, name="res"+str(index)+"_batchnorm2")(x)
    x = Add(name=res_name+"_add")([in_x, x])
    x = Activation("relu", name=res_name+"_relu2")(x)
    return x
