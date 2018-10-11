import numpy as np
import keras
from keras.layers import Reshape, Lambda, Flatten, Activation, Conv2D, Conv2DTranspose, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
import keras.backend as K
import tensorflow as tf
def model_generator(input_shape=(256, 256, 3), input_mask=(256, 256, 1)):
    """
    Architecture of the image completion network
    """
    
    out = Conv2D(64, kernel_size=5, strides=1, padding='same',
                     dilation_rate=(1, 1))(input_shape)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(128, kernel_size=3, strides=2,
                     padding='same', dilation_rate=(1, 1))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(128, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(256, kernel_size=3, strides=2,
                     padding='same', dilation_rate=(1, 1))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(2, 2))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(4, 4))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(8, 8))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(16, 16))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2DTranspose(128, kernel_size=4, strides=2,
                              padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(128, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2DTranspose(64, kernel_size=4, strides=2,
                              padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(32, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(3, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(out)
    out = BatchNormalization()(out)
    out = Activation('sigmoid')(out)
    # x [0] * x [2]: Cut out the region where the mask bit is set from out (make the region other than mask 0)
    # x [1] * (1 - x [2]): Cut out the region where the bit of mask is not set from input_image
    # Merge (add) the above two to make the image replaced only with the output of NN for the mask part
    out = keras.layers.Lambda(lambda x: x[0] * x[2] + x[1] * (1 - x[2]),
                        trainable=False)([out, input_shape, input_mask])
    model = Model([input_shape,input_mask],out)

    return model, out

# The global discriminator network takes the entire image as input, while
# the local discriminator network takes only a small region around the completed area.
def model_discriminator(global_shape=(256, 256, 3), local_shape=(128, 128, 3)):
    def crop_image(img, crop):
        return tf.image.crop_to_bounding_box(img,
                                             crop[1],
                                             crop[0],
                                             crop[3] - crop[1],
                                             crop[2] - crop[0])

    in_pts = Input(shape=(4,), dtype='int32') # [y1,x1,y2,x2]
    cropping = Lambda(lambda x: K.map_fn(lambda y: crop_image(y[0], y[1]), elems=x, dtype=tf.float32),
                      output_shape=local_shape)
    g_img = Input(shape=global_shape)
    l_img = cropping([g_img, in_pts])

    # Local Discriminator
    # Local Discriminator take the point only (mask part ony)
    x_l = Conv2D(64, kernel_size=5, strides=2, padding='same')(l_img)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(128, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(256, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Flatten()(x_l)
    x_l = Dense(1024, activation='relu')(x_l)

    # Global Discriminator
    # Global Discriminator take all image
    x_g = Conv2D(64, kernel_size=5, strides=2, padding='same')(g_img)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(128, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(256, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Flatten()(x_g)
    x_g = Dense(1024, activation='relu')(x_g)

    x = concatenate([x_l, x_g])
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=[g_img, in_pts], outputs=x)
