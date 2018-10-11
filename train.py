import numpy as np
from keras.optimizers import Adadelta,Adam
from keras.layers import merge, Input
from keras.models import Model
from keras.engine.topology import Network
from keras.utils import generic_utils
import keras.backend as K
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from util import process_image, show_image, get_points
from model import model_generator, model_discriminator

filenames = glob('../input/img_align_celeba/img_align_celeba/*')
print(len(filenames))

input_shape = (256, 256, 3)
local_shape = (128, 128, 3)
batch_size = 128
n_epoch = 100
tc = 20
td = 2
alpha = 0.0004

def build_model():
    optimizer = Adadelta()
	
    # build Completion Network model
    org_img = Input(shape=input_shape, dtype='float32')
    mask = Input(shape=(input_shape[0], input_shape[1], 1))
    
    generator,completion_out  = model_generator(org_img, mask)
    completion_model = generator.compile(loss='mse', 
                    optimizer=optimizer)

    # build Discriminator model
    in_pts = Input(shape=(4,), dtype='int32') # [y1,x1,y2,x2]
    discriminator = model_discriminator(input_shape, local_shape)
    d_container = Network(inputs=[org_img, in_pts], outputs=discriminator([org_img, in_pts]))
    d_out = d_container([org_img, in_pts])
    d_model = Model([org_img, in_pts],d_out)
    d_model.compile(loss='binary_crossentropy', 
                    optimizer=optimizer)
    d_container.trainable = False
	
	# build Discriminator & Completion Network models
    all_model = Model([org_img, mask, in_pts],
                      [completion_out, d_out])
    all_model.compile(loss=['mse', 'binary_crossentropy'],
                      loss_weights=[1.0, alpha], optimizer=optimizer)

					  
    X_train = filenames[:5000]
    valid = np.ones((batch_size, 1)) ## label
    fake = np.zeros((batch_size, 1)) ## label

    for n in range(n_epoch):
        progbar = generic_utils.Progbar(len(X_train))
        for i in range(int(len(X_train)//batch_size)):
            X_batch = X_train[i*batch_size:(i+1)*batch_size]
            inputs = np.array([process_image(filename,input_shape[:2]) for filename in X_batch])

            points, masks = get_points(batch_size)
            completion_image = generator.predict([inputs, masks])
            g_loss = 0.0
            d_loss = 0.0
            if n < tc:
                g_loss = generator.train_on_batch([inputs, masks], inputs)
            else:
                d_loss_real = d_model.train_on_batch([inputs, points], valid)
                d_loss_fake = d_model.train_on_batch([completion_image, points], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                if n >= tc + td:
                    g_loss = all_model.train_on_batch([inputs, masks, points],
                                                      [inputs, valid])
                    g_loss = g_loss[0] + alpha * g_loss[1]
            progbar.add(inputs.shape[0], values=[("Epoch", int(n+1)), ("D loss", d_loss), ("G mse", g_loss)])
        # show_image
        show_image(batch_size,n, inputs ,masks ,completion_image)
    # save model
    generator.save("model/generator.h5")
    discriminator.save("model/discriminator.h5")

def main():
    build_model()

if __name__ == "__main__":
    main()
