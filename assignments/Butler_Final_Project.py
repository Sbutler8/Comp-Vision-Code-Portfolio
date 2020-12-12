#!/usr/bin/env python
# coding: utf-8
Author: Samantha Butler
# <h1>Final Project: GAN Implementation</h1>

# In[1]:


import keras
import cv2
from keras import layers
import numpy as np
import os 
import glob
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from PIL import Image
def register_extension(id, extension): Image.EXTENSION[extension.lower()] = id.upper()
Image.register_extension = register_extension
def register_extensions(id, extensions): 
    for extension in extensions: register_extension(id, extension)
Image.register_extensions = register_extensions
get_ipython().run_line_magic('matplotlib', 'inline')


# <h2>The Model</h2>

# <h3>The Generator</h3>

# In[2]:


latent_dim = 32
height = 32
width = 32
channels = 3

generator_input = keras.Input(shape=(latent_dim,))

x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)
generator.summary()


# <h3>The Discriminator</h3>

# In[3]:


discriminator_input = layers.Input(shape=(height, width, channels))

x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

discriminator_optimizer = keras.optimizers.RMSprop(
    lr=0.0008,
    clipvalue=1.0,
    decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy')


# <h3>Adversial Network</h3>

# In[4]:


discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input,gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004,clipvalue=1.0,decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')


# <h2>Preparing Data</h2>

# <h2>Implementing GAN</h2>

# In[ ]:


from keras.preprocessing import image

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
IMAGE_CHANNELS = 3

# img = cv2.imread('/home/students/samantha.butler/Final/images/cropped/d817b596-b932-42ef-97de-b939549cc542.jpg')
# plt.imshow(img)

def getImages(directory):
    images = [];
    for directoryPath, directoryNames, fileNames in os.walk(directory):
        for fileName in fileNames:
            imageFile = os.path.join(directoryPath, fileName)
            img = cv2.imread(imageFile)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             plt.imshow(img)
#             print(img.shape)
            images.append(img)
            
    
    return images

def make_square(img):
    plt.imshow(img)
    cols = img.shape[0]
    rows = img.shape[1]
    print("cols: ", cols, "rows: ", rows)
   
    if rows>cols:
        pad = (rows-cols)/2
        img = img.crop((pad,0,cols,cols))
    else:
        pad = (cols-rows)/2
        img = img.crop((0,pad,rows,rows))
    
    return img

def normalizedTrainingData(images):  
    x_train = []
    
    for img in images:
        img = make_square(img)
        img = img.resize((IMAGE_WIDTH,IMAGE_HEIGHT),Image.ANTIALIAS)
        x_train.append(np.asarray(img))
    return x_train


# In[ ]:


directory = '/home/students/samantha.butler/Final/images/cropped'
images = getImages(directory)
# print(images)


# In[ ]:


x_train = normalizedTrainingData(images)
print(x_train)


# In[ ]:


# Could not get my x_train data in the proper format unfortunately and so I could not run the full GAN network. 
# With the correction of the previous cell all that is needed to do is insert that x_train data into my GAN function.
# The output will store the generated weights of the images and a saved generated image in the '/gan_images' directory

def GAN(x_train, iterations, batch_size, save_dir):
    start = 0
    for step in range(iterations):
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

        generated_images = generator.predict(random_latent_vectors)

        stop = start + batch_size
        real_images = x_train[start: stop]
        combined_images = np.concatenate([generated_images, real_images])

        labels = np.concatenate([np.ones((batch_size, 1)),
                                         np.zeros((batch_size, 1))])
        labels += 0.05 *np.random.random(labels.shape)

        d_loss = discriminator.train_on_batch(combined_images, labels)

        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

        misleading_targets = np.zeros((batch_size, 1))

        a_loss = gan.train_on_batch(random_latent_vectors, misleading, targets)

        start += batch_size
        if start > len(x_train) - batch_size:
            start = 0

        if step % 100 == 0:
            gan.save_weights('gan.h5')

            print('discriminator loss: ', d_loss)
            print('adversarial_loss: ', a_loss)

            # Test saving one generated image of myself
            img =  image.array_to_img(generated_images[0] * 255., scale = False)
            img.save(os.path.join(save_dir, 'generated_sami' + str(step) + '.png'))
            
iterations = 10000
batch_size = 20
save_dir = '/gan_images'
            
GAN(images[0][0], iterations, batch_size, save_dir)


# In[ ]:




