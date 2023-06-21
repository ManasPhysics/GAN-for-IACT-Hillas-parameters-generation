#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:04:58 2022

@author: manas
"""

# train a generative adversarial network on a one-dimensional function
from numpy import hstack,vstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
import keras 
from keras.models import Sequential
from keras.layers import Dense,Dropout,LeakyReLU
from keras.utils import plot_model
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math 


#data=pd.read_csv("DataFile/Gamma_TACTIC_Zen05_Hillas.dat",sep='\s+')
columns=['Energy','Length', 'Width', 'Dist', 'Alpha', 'F2', 'Size','Asym']
data=pd.read_csv("DataFile/HillasGamma_Zen05_PE18.dat",sep='\s+',usecols=columns)

data=data[(data.Alpha<89.0) & (data.Alpha>0.0)]
data.Alpha=-np.log(3.14*data.Alpha/180.0) 
data.Energy=np.log10(data.Energy)
data.Size=np.log10(data.Size)
# Normalize the input data(i.e. mean=0,std=1.0) 
data=data[data.Asym>0.0]   ### assending order of size 
#data = data.('Asym', ascending=True)
data=(data-data.mean(axis=0))/np.std(data)
########## Trimming the data ##################################################################################################
    #### Trimming for Gamma_GAN scatter plot
    #data=data[(data.Width<3.8) & (data.Asym<1.23) &(data.Length<3.33) & (data.Asym>-2)]
    #### Trimming for Proton_GAN scatter plot ####
    #data=data[(data.Width<4.33) & (data.Asym<2.0) &(data.Length<4.60) & (data.Asym>-2.0) & (data.Alpha<4.22) &(data.F2<3.78)]


## Nos of data #####################
NL=len(data)
#### DROPOUT percentage ###########
DROP=0.0
latent_dim = 16 # For single variable 5 dim is okay

# Learning rate selection 
optD = keras.optimizers.Adam(learning_rate=0.03)
optG = keras.optimizers.Adam(learning_rate=0.001)

#####Training Epochs############
EPOCHS=1000
INPUTS=len(data.columns)


# define the standalone discriminator model

def define_discriminator(n_inputs=INPUTS):
    model = Sequential()
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(32))
    model.add(Dropout(DROP))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(16))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(INPUTS))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer=optD , metrics=['accuracy'])
    return model
 
# define the standalone generator model
 
def define_generator(latent_dim, n_outputs=INPUTS):
    model = Sequential()
    model.add(Dense(64, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(16))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(n_outputs, activation='linear'))
    return model
 
 
 
# define the combined generator and discriminator model, for updating the generator



def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the discriminator
    model.add(discriminator)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer=optG ) # origibnally ,  optimizer='adam'
    return model
 
    
# generate n real samples with class labels
def generate_real_samples(n):
    # generate inputs in [-0.5, 0.5]
    X=np.array(data.sample(n))
    
    # My old method, updated on 08.08.22
    # stack arrays
    ###### WE should use n as a paremeter here ########
    # generate class labels
    y = ones((n, 1))
    return X, y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
    # generate points in the latent space
    x_input = randn(latent_dim * n)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)
    return x_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = zeros((n, 1))
    return X, y
 
    
 
    
 
    
 #### Ths section is important #########################################################################
    
g_loss_hist,g_acc_hist,d_loss_R_hist,d_loss_F_hist,d_acc_R_hist,d_acc_F_hist=[],[],[],[],[],[]
# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=EPOCHS, n_batch=2000, n_eval=2000 , verbose=0):
    # determine half the size of one batch, for updating the discriminator
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    
    for i in range(n_epochs):
        # prepare real samples 
        x_real, y_real = generate_real_samples(half_batch)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator
        
        d_loss_R, d_acc_R=d_model.train_on_batch(x_real, y_real)
        d_loss_F, d_acc_F=d_model.train_on_batch(x_fake, y_fake)
        # prepare points in latent space as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator's error
        g_loss=gan_model.train_on_batch(x_gan, y_gan) 
        # evaluate the model every n_eval epochs
        if (i+1) % n_eval == 0:
            summarize_performance(i, g_model, d_model, latent_dim)
        
        g_loss_hist.append(g_loss)
        d_loss_R_hist.append(d_loss_R)
        d_loss_F_hist.append(d_loss_F)
        d_acc_R_hist.append(d_acc_R)
        d_acc_F_hist.append(d_acc_F)
# size of the latent space
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# train model
train(generator, discriminator, gan_model, latent_dim)



"""
#### This module is for loss and accuray plot ######

fig1 = plt.figure("Figure 1")
#plt.plot(g_loss_hist,linewidth=0.70,color='green',label='Generator loss')
plt.plot(1.0*np.array(d_loss_R_hist),linewidth=1.0,color='red',label=' Disc. loss Real')
plt.plot(1.0*np.array(d_loss_F_hist),linewidth=1.0,color='blue',label='Disc. loss Fake')
plt.xlabel(r'$\mathbf{Epochs}$',fontsize=12)
plt.ylabel('Discriminator loss',fontsize=12,fontweight='bold')
plt.xlim(0,EPOCHS)
plt.ylim(0.0,1.2)
plt.xlim(0,EPOCHS )
plt.hlines(0.693,0,EPOCHS,ls='dashed',color='black',label='Loss=log2')
plt.legend()



fig2 = plt.figure("Figure 2")
plt.plot(np.array(d_acc_R_hist),linewidth=1.0,color='red',label='Disc. accu Real')
plt.plot(np.array(d_acc_F_hist),linewidth=1.0,color='blue',label='Disc. accu Fake')
plt.xlabel(r'$\mathbf{Epochs}$',fontsize=12)
plt.ylabel(r'$\mathbf{Accuracy}$',fontsize=12)
plt.ylim(0,1.1)
plt.xlim(0,EPOCHS)
plt.hlines(0.5,0,1000,ls='dashed',color='black',label='Accuracy=50%')
plt.legend() 
##### Loss and accuracy plot ends here ###########
"""


