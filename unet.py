import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dropout, ReLU,PReLU,ELU, LeakyReLU, Concatenate, Activation, Lambda, BatchNormalization, SpatialDropout2D, Flatten, Dense, Reshape, Conv1D
from keras.utils import multi_gpu_model
from keras.optimizers import Adam, Adadelta, Adamax, SGD
from keras import backend as K
from keras import regularizers
import tensorflow as tf

class UNet(object):
    
    def __init__(self,
                 dims,
                 kernels = [3,3],
                 strides = [2],
                 depth = [32,32],
                 #init_depth = 128,
                 #depth_scale = 2,
                 dropout_rate=.2,
                 load_dir = None,
                 gpus = 1,
                 print_summary=False
                ):
        self.dims = dims
        self.kernels = kernels
        self.strides = strides
        self.depth = depth
        self.dropout_rate=dropout_rate
        self.gpus = gpus
        self.load_dir = load_dir
        self.print_summary = print_summary
        self.Conv, self.Cropping, self.UpS, self.MaxPool = self.set_conv_ups_crop_pool()
        
        if load_dir is not None:
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
            
    
    def down_block(self, feature_map, depth, kernel, stride, final=False):
        reg = regularizers.l2(.1)
        x = feature_map       
        x = self.Conv(depth, kernel, padding='same', kernel_regularizer=reg)(x)
        x = Dropout(self.dropout_rate)(x, training = True)
        x = PReLU()(x)
        x = self.Conv(depth, kernel, padding='same', kernel_regularizer=reg)(x)
        x = Dropout(self.dropout_rate)(x, training = True)
        x = BatchNormalization(momentum=0.9)(x)
        x = PReLU()(x)
        x = BatchNormalization(momentum=0.9)(x)
        x_pass=x
        if final is not True:
            x = self.Conv(depth,stride,strides=stride,padding='valid', kernel_regularizer=reg)(x)
            x = Dropout(self.dropout_rate)(x, training = True)
            x = PReLU()(x)
            return (x,x_pass)
        else:
            return x
        
    def up_block(self, feature_map, copy_map, depth, kernel, stride):
        reg = regularizers.l2(.1)
        x = feature_map
        x = self.UpS(stride)(x)
        x = self.Conv(depth, stride, padding='same', kernel_regularizer=reg)(x)
        x = Dropout(self.dropout_rate)(x, training = True)
        x = PReLU()(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Concatenate()([x,copy_map])
        x = self.Conv(depth, kernel, padding='same', kernel_regularizer=reg)(x)
        x = Dropout(self.dropout_rate)(x, training = True)
        x = PReLU()(x)
        x = self.Conv(depth, kernel, padding='same', kernel_regularizer=reg)(x)
        x = Dropout(self.dropout_rate)(x, training = True)
        x = PReLU()(x)
        return x
        
    def build_unet(self):
        
        map_copies = []
        
        input_img = Input(shape=self.dims)
        x = input_img  
        
        
        for k,s,d in zip(self.kernels,self.strides,self.depth):
            x, copy = self.down_block(x,d,k,s)
            map_copies.append(copy)
        
        x = self.down_block(x,self.depth[-1],self.kernels[-1],1,final=True)
        
        for k,s,d in zip(self.kernels[:-1][::-1],self.strides[::-1], self.depth[:-1][::-1]):
            x = self.up_block(x, map_copies.pop(), d, k, s)
            
        output_img = self.Conv(1,1,padding='same')(x)
        x=Reshape((256,256))(output_img)
        x = Lambda(power1D)(x)
        x = Reshape((128,1))(x)
        x = Conv1D(32,3,padding='same')(x)
        x = Dropout(self.dropout_rate)(x, training = True)
        x = PReLU()(x)
        x = Conv1D(1,3,padding='same')(x)
        output_power = Reshape((128,))(x)
        if self.gpus <=1:
            UNet = Model(inputs=input_img,
                        outputs=[output_img,output_power],
                        name = 'Unet Model')
            if self.print_summary: UNet.summary()
        else:
            with tf.device("/cpu:0"):
                UNet = Model(inputs=input_img,
                            outputs=[output_img,output_power])
            if self.print_summary: UNet.summary()
            UNet = multi_gpu_model(UNet,gpus=self.gpus)
        UNet.compile(loss=[self.mse_map,self.mse_power],#metrics=[self.mse_map,self.mse_power,'mape'],
                    optimizer=Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, clipnorm=10.))

#         UNet.compile(loss=self.loss,metrics=[self.mse,self.mape],
#                     optimizer=Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, clipnorm=10.))

        return UNet
    
    def loss(self, y_true, y_pred):
        mean = K.reshape(y_pred[:,:,:,0],[-1,256,256,1])
        var = K.reshape(y_pred[:,:,:,1],[-1,256,256,1])
        diff_sq = K.square(y_true-mean)
        return K.mean(K.sum(.5*diff_sq*K.exp(-var)+.5*var,axis=np.arange(1, len(diff_sq.shape))))
    
    def mse(self, y_true, y_pred):
        mean = K.reshape(y_pred[:,:,:,0],[-1,256,256,1])
        diff_sq = K.square(y_true-mean)
        return K.mean(diff_sq)
    
    def mape(self, y_true, y_pred):
        mean = K.reshape(y_pred[:,:,:,0],[-1,256,256,1])
        diff_abs = K.abs((y_true-mean)/(y_true+K.epsilon())*100)
        return K.mean(diff_abs)
    
    def mse_power(self, y_true, y_pred):
        power_diff_sq = K.square(y_true-y_pred)
        return K.mean(power_diff_sq)
    
    def mse_map(self, y_true, y_pred):
        diff_sq = K.square(y_true-y_pred)
        return K.mean(diff_sq)
    
    def mse_map_power(self, y_true, y_pred):
        return self.mse_map(y_true, y_pred) + self.mse_power(y_true, y_pred)


def power2D(x):
    x = tf.spectral.fft2d(tf.cast(x,dtype=tf.complex64))
    x = tf.cast(x,dtype=tf.complex64)
    xl,xu = tf.split(x,2,axis=1)
    xll,xlr = tf.split(xl,2,axis=2)
    xul,xur = tf.split(xu,2,axis=2)
    xu = tf.concat([xlr,xll],axis=2)
    xl = tf.concat([xur,xul],axis=2)
    x=tf.concat([xl,xu],axis=1)
    x = tf.abs(x)
    return tf.square(x)
     
class AZAverage(object):
    
    def __init__(self,size):
        self.size = size
        x,y = np.meshgrid(np.arange(size),np.arange(size))
        R = np.sqrt((x-size/2)**2+(y-size/2)**2)
        masks = np.array(list(map(lambda r : (R >= r-.5) & (R < r+.5),np.arange(1,int(size/2+1),1))))
        norm = np.sum(masks,axis=(1,2),keepdims=True)
        masks=masks/norm
        n=len(masks)
        self.big_mask = tf.reshape(tf.cast(masks,dtype=tf.float32),(1,n,size,size))
        
    def __call__(self,v):
        v=tf.reshape(v,(-1,1,self.size,self.size))
        return tf.reduce_sum(tf.reduce_sum(tf.multiply(self.big_mask,v),axis=3),axis=2)

az_average = AZAverage(256)
    
def power1D(x):
    x = power2D(x)
    az_avg = az_average(x)
    ell=np.arange(int(az_avg.shape[1]))*9
    return tf.multiply(az_avg,tf.reshape(tf.cast(ell*(ell+1)/2/np.pi/1e10,dtype=tf.float32),(1,-1)))