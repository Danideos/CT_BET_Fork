import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate


def Unet(image_shape, initializer='glorot_uniform', nb_classes=2):
    inputs = Input(shape=image_shape)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', kernel_initializer=initializer)(drop5), drop4], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(conv6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer=initializer)(conv6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(conv7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer=initializer)(conv7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(conv8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer=initializer)(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(conv9)

    conv10 = Conv2D(nb_classes, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model
