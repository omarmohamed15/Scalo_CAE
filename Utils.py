from matplotlib.colors import ListedColormap
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift
from scipy import ndimage, misc
import obspy
from obspy.imaging.cm import obspy_sequential
import obspy
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt
from sklearn import preprocessing
from keras.layers import Input, Dense, Dropout, concatenate, UpSampling1D, Flatten, MaxPooling1D, MaxPooling2D,UpSampling2D,Conv2DTranspose, BatchNormalization, average, Conv1D, Add
from keras.models import Model
import scipy.io
from keras import optimizers
from keras import backend as K
from tensorflow.python.keras import backend as KK
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import h5py
from sklearn import preprocessing
import tensorflow as tf
import numbers
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
from math import*
import numpy as np
import matplotlib.pyplot as plt

import obspy
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, Activation


import tensorflow as tf
import random as rn

def seisplot_wig1(s, inc=1, scale=0.8, lw=1, highlight=False, lightstep=10, 
                 figsize=(7.5,6)):
    
    nt = s.shape[0]
    xmax = s.shape[1]
    #vmax = np.max(s1)
    t = np.arange(nt)
    
    fig, ax = plt.subplots(figsize=(9,6))
    for i in np.arange(0,xmax,inc):
        x1 = scale * s[:,i] + i
        
        ax.plot(x1, t, 'k', lw=lw)
        ax.fill_betweenx(t, x1, x2=i, where=x1>=i, facecolor='k', 
                          interpolate=True)
        if highlight==True:
            if i % lightstep == 0:
                ax.plot(x1, t, 'r', lw=lw*2)
                ax.fill_betweenx(t, x1, x2=i, where=x1>=i, facecolor='r', 
                              interpolate=True)
            if i==int(xmax/lightstep)*lightstep:
                ax.plot(x1, t, 'r', lw=lw*2, label='Training traces')
                ax.legend(loc='upper right' , fontsize=12)
    #ax.invert_yaxis()
    #ax.set_xlim(-1,xmax)
    #ax.set_ylim(nt,0)
    #ax.set_ylabel('Time samples', fontsize=13)
    #ax.set_xlabel('Traces', fontsize=13)
    fig.tight_layout()
    
    return fig, ax


def predict_label(s11,encoder,scal,nf,D4):
    labelproposed = np.zeros((np.shape(s11)[0]))
    clabelf = np.zeros((np.shape(s11)[0] , np.shape(s11)[1]))
    for ix in range(0,np.shape(s11)[0]):
        scalre = scal[ix]
        scal1 = np.reshape(scalre,(1,np.shape(s11)[1],nf,1))

        outenc = encoder.predict(scal1)**2
        outenc = np.reshape(outenc,(np.shape(s11)[1],D4))

        #outenc = autoencoder.predict(scal1)**2
        #outenc = np.reshape(outenc, (np.shape(s11)[1],nf))


        var =[]
        indout=[]
        le=8

        for io in range(0,D4):
            xz = outenc[:,io]
            var.append(np.std(xz))
        varord = np.flip(np.sort(var), axis =-1)

        for iuy in range(0,le):
            indout.append(np.where(var == varord[iuy])[0][0])

        outenc1 = outenc[:,indout] 
        clabelpre = np.zeros_like(outenc1)
        for iux in range(0,le):

            iuxx = indout[iux]
            fet = outenc[:,iuxx]
            fet = fet / np.max(np.abs(fet))


            me  = np.mean(fet)
            clabelxx = np.zeros_like(fet)
            indlarge = np.where(fet>1*me)[0]
            clabelxx[indlarge] =1
            clabelpre[:,iux] = clabelxx
            #clabelpre[:,iux] = fet


        cluster = KMeans( n_clusters=2, random_state=0).fit(clabelpre)
        clabel = cluster.labels_
        acp = [len(np.where(clabel==1)[0]),len(np.where(clabel==0)[0])]
        ap = np.min(acp)
        ak = np.where(acp == ap)
        if ak[0][0] ==1:
            clabel = (clabel-1)*-1
        try:
            labelproposed[ix] = np.max([np.where(clabel==1)[0][0],np.where(clabel==0)[0][0]])
            labelproposed[ix] = np.where(clabel==1)[0][0]
            clabelf[ix,0:len(clabel)] = clabel
        except:
            labelproposed[ix] =0 
            clabelf[ix,0:len(clabel)] = clabel
            
    return clabelf,labelproposed
    