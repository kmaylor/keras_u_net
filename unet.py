import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dropout, ReLU, Concatenate
from keras.utils import multi_gpu_model
from keras import backend as K
import tensorflow as tf

class UNet(object):
    
    def __init__(self,
                 dims,
                 kernels = [3,3],
                 strides = [2],
                 init_depth = 64,
                 depth_scale = 2,
                 load_dir = None,
                 gpus = 1,
                ):
        self.dims = dims
        self.kernels = kernels
        self.strides = strides
        self.init_depth = init_depth
        self.depth_scale = depth_scale
        self.gpus = gpus
        self.load_dir = load_dir
        self.Conv, self.Cropping, self.UpS, self.MaxPool = self.set_conv_ups_crop_pool()
        
        if load_dir != None:
            print('Loading Previous State')
            self.UNet = load_model(load_dir)
        else:
            self.UNet = self.build_unet()
        
    def set_conv_ups_crop_pool(self):
        d = len(self.dims)-1
        if d == 1:
            from keras.layers import Conv1D, Cropping1D, UpSampling1D, MaxPooling1D
            return (Conv1D, Cropping1D, UpSampling1D, MaxPooling1D)
        elif d == 2:
            from keras.layers import Conv2D, Cropping2D, UpSampling2D, MaxPooling2D
            return (Conv2D, Cropping2D, UpSampling2D, MaxPooling2D)
        else:
            from keras.layers import Conv3D, Cropping3D, UpSampling3D, MaxPooling3D
            return (Conv3D, Cropping3D, UpSampling3D, MaxPooling3D)
            
    
    def down_block(self, feature_map, depth, kernel, stride):
        x = self.Conv(depth, kernel, padding='same')(feature_map)
        x = ReLU()(x)
        x = self.Conv(depth, kernel, padding='same')(x)
        x = ReLU()(x)
        if stride != 1: 
            down = self.MaxPool(pool_size=stride, padding='valid')(x)
            return (down,x)
        else:
            return x
        
    def up_block(self, feature_map, copy_map, depth, kernel, stride):
        x = self.UpS(stride)(feature_map)
        x = self.Conv(depth, 2, padding='same')(x)
        x = Concatenate()([x,copy_map])
        x = self.Conv(depth, kernel, padding='same')(x)
        x = ReLU()(x)
        x = self.Conv(depth, kernel, padding='same')(x)
        x = ReLU()(x)
        return x
        
    def build_unet(self):
        
        map_copies = []
        depth = self.init_depth
        
        input_img = Input(shape=self.dims)
        x = input_img    
        
        for k,s in zip(self.kernels,self.strides):
            x, copy = self.down_block(x,depth,k,s)
            map_copies.append(copy)
            depth*=self.depth_scale
        
        x = self.down_block(x,depth,self.kernels[-1],1)

        for k,s in zip(self.kernels[:-1][::-1],self.strides[::-1]):
            depth=int(depth/self.depth_scale)
            x = self.up_block(x, map_copies.pop(), depth, k, s)
            
        output_img = self.Conv(1,1,padding='same')(x)
        
        
        if self.gpus <=1:
            UNet = Model(inputs=input_img,
                        outputs=output_img,
                        name = 'Unet Model')
        else:
            with tf.device("/cpu:0"):
                UNet = Model(inputs=input_img,
                            outputs=output_img,
                            name = 'Unet Model')
            UNet = multi_gpu_model(UNet,gpus=self.gpus)
        UNet.compile(loss='mse',
                    optimizer='adam')

        UNet.summary()
        return UNet
        
        
            
