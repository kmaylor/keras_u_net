import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dropout, ReLU, ELU, LeakyReLU, Concatenate, Activation, Lambda, BatchNormalization, Reshape, Add, Conv1D, PReLU, Activation, GlobalAveragePooling2D, Dense, Multiply, GlobalAveragePooling1D
from keras.utils import multi_gpu_model
from keras.optimizers import Adam, Adadelta, Adamax, SGD
from keras import backend as K
from keras import regularizers
import tensorflow as tf
from tensorflow_power_spectrum import PowerSpectrum

ps = PowerSpectrum(image_size=256, scale=1e-11)
    
class ResUNet(object):
    
    def __init__(self,
                 dims,
                 kernels,
                 strides,
                 depth,
                 dropout_rates,
                 load_dir = None,
                 gpus = 1,
                 print_summary=False
                ):
        self.dims = dims
        self.kernels = kernels
        self.strides = strides
        self.depth = depth
        self.dropout_rates=dropout_rates
        self.gpus = gpus
        self.load_dir = load_dir
        self.print_summary = print_summary
        self.Conv, self.UpS = self.set_conv_ups_crop_pool()
        
        if load_dir is not None:
            print('Loading Previous State')
            self.UNet = load_model(load_dir)
        else:
            self.UNet = self.build_unet()
        
    def set_conv_ups_crop_pool(self):
        d = len(self.dims)-1
        if d == 1:
            from keras.layers import Conv1D, UpSampling1D
            return (Conv1D, UpSampling1D)
        elif d == 2:
            from keras.layers import Conv2D, UpSampling2D
            return (Conv2D, UpSampling2D)
        else:
            from keras.layers import Conv3D, UpSampling3D
            return (Conv3D, UpSampling3D)
        
    def se_block(self,x):
        s = GlobalAveragePooling2D()(x)
        s = Dense(int(x.shape[-1]//2), activation="relu")(s)
        s = Dense(int(x.shape[-1]), activation="sigmoid")(s)
        return Multiply()([x, s])
            
    
    def down_block(self, feature_map, depth, kernel, stride, drop_rate, bridge=False):
        reg = regularizers.l2(.1)
        x = feature_map
        x = self.Conv(depth, kernel, strides=stride, padding='same', kernel_regularizer=reg)(x)
        x = Dropout(drop_rate)(x, training = True)
        x = LeakyReLU(.2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = self.Conv(depth, kernel, padding='same', kernel_regularizer=reg)(x)
        x = Dropout(drop_rate)(x, training = True)
        x = LeakyReLU(.2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = self.se_block(x)
        if not bridge: x = Add()([x,self.Conv(depth, kernel, strides=stride, padding='same', kernel_regularizer=reg)(feature_map)])
        return x
        
    def up_block(self, feature_map, copy_map, depth, kernel, stride, drop_rate):
        reg = regularizers.l2(.1)
        x = feature_map
        x = self.UpS(stride)(x)
        x_c = Concatenate()([x,copy_map])
        x = x_c
        x = self.Conv(depth, kernel, padding='same', kernel_regularizer=reg)(x)
        x = Dropout(drop_rate)(x, training = True)
        x = LeakyReLU(.2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = self.Conv(depth, kernel, padding='same', kernel_regularizer=reg)(x)
        x = Dropout(drop_rate)(x, training = True)
        x = LeakyReLU(.2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = self.se_block(x)
        x = Add()([x,self.Conv(depth, kernel, padding='same', kernel_regularizer=reg)(x_c)])
        return x
        
    def build_unet(self):
        
        connections = []
        
        input_img = Input(shape=self.dims)
        x = input_img  
        
        i=0
        bridge = False
        for k,s,d,dr in zip(self.kernels,self.strides,self.depth,self.dropout_rates):
            if i == len(self.depth)-1: bridge=True
            x = self.down_block(x,d,k,s,dr, bridge=bridge)
            if not bridge: connections.append(x)
            i+=1
            
        for k,s,d,dr in zip(self.kernels[:-1][::-1],self.strides[::-1], self.depth[:-1][::-1], self.dropout_rates[:-1][::-1]):
            x = self.up_block(x, connections.pop(), d, k, s, dr)
            
        output_img = self.Conv(1,1,padding='same')(x)
        x = Lambda(lambda v: v[:,:,:,0])(output_img)
        x=Reshape((256,256))(x)
        #output_power = Lambda(lambda x: ps.power1D(x))(x)
        x = Lambda(lambda x: ps.power1D(x))(x)
        xr = Reshape((128,1))(x)
        x = Conv1D(64,5,padding='same')(xr)
        x = Dropout(.1)(x, training = True)
        x = LeakyReLU(.2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv1D(64,5,padding='same')(xr)
        x = Dropout(.1)(x, training = True)
        x = LeakyReLU(.2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        s = GlobalAveragePooling1D()(x)
        s = Dense(int(x.shape[-1]//2), activation="relu")(s)
        s = Dense(int(x.shape[-1]), activation="sigmoid")(s)
        x = Multiply()([x, s])
        x = Conv1D(1,1,padding='same')(x)
        x = Add()([xr,x])
        output_power = Reshape((128,))(x)
        
        if self.gpus <=1:
            UNet = Model(inputs=input_img,
                        outputs=[output_img,output_power],
                        name = 'Unet Model')
            if self.print_summary: UNet.summary()
        else:
            with tf.device("/cpu:0"):
                UNet = Model(inputs=input_img,
                            #outputs=output_img)
                            outputs=[output_img,output_power])
            if self.print_summary: UNet.summary()
            UNet = multi_gpu_model(UNet,gpus=self.gpus)
        UNet.compile(loss=['mse',self.mse_power],
                    optimizer=Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, clipnorm=10))

#         UNet.compile(loss=[self.like,self.plike],#metrics=[self.mse,self.mape],
#                     optimizer=Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, clipnorm=100.))

        return UNet
    
    def like(self, y_true, y_pred):
        mean = K.reshape(y_pred[:,:,:,0],[-1,256,256,1])
        var = K.reshape(y_pred[:,:,:,1],[-1,256,256,1])
        diff_sq = K.square(y_true-mean)
        return K.mean(K.sum(.5*diff_sq*K.exp(-var)+.5*var,axis=np.arange(1, len(diff_sq.shape))))
    
    def plike(self, y_true, y_pred):
        pc = lambda v: ps.power1D(v)
        y_true=pc(K.reshape(y_true,(-1,256,256)))
        mean = K.reshape(y_pred[:,:,0],[-1,128])
        var = K.reshape(y_pred[:,:,1],[-1,128])
        diff_sq = K.square(y_true-mean)
        return K.mean(K.sum(.5*diff_sq*K.exp(-var)+.5*var,axis=np.arange(1, len(diff_sq.shape))))
    
    
    def mse(self, y_true, y_pred):
        mean = K.reshape(y_pred[:,:,:,0],[-1,128,128,1])
        diff_sq = K.square(y_true-mean)
        return K.mean(diff_sq)
    
    def mape(self, y_true, y_pred):
        mean = K.reshape(y_pred[:,:,:,0],[-1,128,128,1])
        diff_abs = K.abs((y_true-mean)/(y_true+K.epsilon())*100)
        return K.mean(diff_abs)
    
    def mse_power(self,y_true, y_pred):
        pc = lambda v: ps.power1D(v)
        y_true = pc(K.reshape(y_true,(-1,256,256)))
        power_diff_sq = K.square(y_true-y_pred)
        return K.mean(power_diff_sq)
    
    def mse_map(self, y_true, y_pred):
        diff_sq = K.square(y_true-y_pred)
        return K.mean(diff_sq)
    
#     def mse_map_power(self, y_true, y_pred):
#         return self.mse_map(y_true, y_pred) + self.mse_power(y_true, y_pred)


