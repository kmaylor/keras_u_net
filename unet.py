import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import LeakyReLU, Dropout, Lambda, ReLU
from keras.layers import BatchNormalization, Concatenate
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras import backend as K
import tensorflow as tf

class UNet(object):
    
    def __init__():
        self.Conv, self.UpS, self.Cropping, self.MaxPool = self.set_convolution()
        
    def set_conv_ups_crop(self):
        d = len(self.dims)
        if d == 1:
            from keras.layers import Conv1D, Cropping1D, UpSampling1D, MaxPooling1D
            return (Conv1D, Cropping1D, UpSampling1D, MaxPooling1D)
        elif d == 2:
            from keras.layers import Conv2D, Cropping2D, UpSampling2D, MaxPooling2D
            return (Conv2D, Cropping2D, UpSampling2D, MaxPooling2D)
        else:
            from keras.layers import Conv3D, Cropping3D, UpSampling3D, MaxPooling3D
            return (Conv3D, Cropping3D, UpSampling3D, MaxPooling3D)
            
    
    def down_block(feature_map, depth, kernel, stride):
        x = self.Conv(depth,kernel,1)(feature_map)
        x = ReLU(x)
        x = self.Conv(depth,kernel,1)(x)
        x = ReLU(x)
        down = self.MaxPool(pool_size=2)(x)
        return (down,x)
        
    def up_block(feature_map, copy_map, depth, kernel, stride):
        x = self.UpS(size=stride)(feature_map)
        x = self.Conv
        x = Concatenate(x,copy_map)
        