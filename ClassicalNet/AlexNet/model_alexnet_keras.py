# 用keras快速搭建AlexNet网络

from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, \
    Dense, BatchNormalization, Activation, MaxPooling2D, concatenate, Dropout, Flatten

nb_classes = 10
img_rows = 227
img_cols = 227
img_channels = 227


def construct_net(img_input, classes=10):
    x = Conv2D(96, (11, 11), strides=(4, 4), padding='same',
               activation='relu', kernel_initializer='uniform')(img_input)  # valid
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Conv2D(256, (5, 5), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='uniform')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='uniform')(x)

    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='uniform')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='uniform')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(classes, activation='softmax')(x)

    return output


img_input = Input(shape=(img_rows, img_cols, img_channels))
output = construct_net(img_input)
model = Model(img_input, output)
