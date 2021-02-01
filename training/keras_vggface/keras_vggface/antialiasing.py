from keras import backend as K
from keras.layers import Layer, ZeroPadding2D, InputSpec
import tensorflow as tf
import numpy as np
from math import ceil

MAX_SUPPORTED_KERNEL_SIZE = 7

class BlurPool(Layer):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0, **kwargs):
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]*2
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        #print(self.pad_sizes)
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
        if self.pad_sizes[0]==0 and self.pad_sizes[1]==0:
            self.pad = lambda x: x
        else:
            print("Pad sizes: %s"%str(self.pad_sizes))
            self.pad = get_pad_layer(pad_type)(self.pad_sizes)
        super(BlurPool, self).__init__(**kwargs)

    def build(self, input_shape):
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
        else:
            raise

        filt_np = a[:,None]*a[None,:]
        filt_np = filt_np / np.sum(filt_np)
        
        if K.image_data_format()=='channels_first':
            filt_np = np.repeat(filt_np[None,:,:], input_shape[0], 0)
        else:
            filt_np = np.repeat(filt_np[:,:,None], input_shape[-1], -1)
        filt_np = np.expand_dims(filt_np, -1)
        #print(filt_np.shape)
        self.filt = K.constant( filt_np, name='filt' )
        
        super(BlurPool, self).build(input_shape)

    def call(self, inp):
        if self.stride==1: # Even if the filter is >1, do not blur if there is no downsampling
            return inp

        if(self.filt_size==1):
            if(self.pad_off==0):
                x = inp
            else:
                x = self.pad(inp)

            if K.image_data_format()=='channels_first':
                return x[:,:,::self.stride,::self.stride]
            else:
                return x[:,::self.stride,::self.stride,:]
        else:
            #print ("shape after pad: %s" % str(self.pad.compute_output_shape(inp.shape)))
            return K.depthwise_conv2d(self.pad(inp), self.filt, 
                        strides=(self.stride,self.stride))

    def compute_output_shape(self, input_shape):
        s = list(input_shape)
        t = 1 if K.image_data_format()=='channels_first' else 0
        s[t+1]= int(ceil(s[t+1]/self.stride))
        s[t+2]= int(ceil(s[t+2]/self.stride))
        #print("Shape out for %s: %s" %(self.name,str(s)))
        return tuple(s)


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + self.padding[0]+ self.padding[1], s[2] + self.padding[2] + self.padding[3], s[3])

    def call(self, x, mask=None):
        p = self.padding
        return tf.pad(x, [[0,0], [p[0],p[1]], [p[2],p[3]], [0,0] ], 'REFLECT')


def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = ReflectionPadding2D
    elif(pad_type=='zero'):
        PadLayer = ZeroPadding2D
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


def main_test():
    import cv2
    import keras
    im = cv2.imread('lenna.jpg')
    im = im[:-1, :-1]

    model = keras.Sequential()
    #model.add( keras.layers.Conv2D(filters=16, kernel_size=(3,3)) )
    model.add( BlurPool(filt_size=2) )
    model.build( tuple([1]+list(im.shape)) )
    model.summary()

    out = model.predict(np.expand_dims(im, 0))

    out = np.squeeze(out).astype(np.uint8)
    print(out.shape, out.dtype)
    '''
    showim = np.zeros( (im.shape[0], 2*im.shape[1], 3), dtype=np.uint8)
    showim[:, 0:im.shape[1], :] = im
    showim[0:out.shape[0], im.shape[1]:im.shape[1]+out.shape[1], :] = out
    cv2.imshow('showim', showim)
    cv2.waitKey(0)
    '''

if '__main__' == __name__:
    main_test()

from keras.layers import Conv2D, MaxPooling2D
class LPFConv2D():
    def __init__(self, filters, kernel_size,
        pad_type='reflect', lpf_size=3, strides=(1,1),
        channels=None, pad_off=0, **kwargs):
        if isinstance(strides, int):
            strides=(strides,strides)
        if strides[0] != strides[1]:
            raise 
        if strides[0] > MAX_SUPPORTED_KERNEL_SIZE:
            raise 
        
        if strides[0] > 1 and lpf_size>0:
            if 'name' in kwargs:
                blurpool_name = kwargs['name']+'_blurpool'
            else:
                blurpool_name = None
            self.blurpool = BlurPool(pad_type=pad_type, filt_size=lpf_size,
                            stride=strides[0], channels=channels, 
                            pad_off=pad_off, name=blurpool_name)
        else:
            self.blurpool = None
        self.conv2d = Conv2D(filters, kernel_size, strides=strides if self.blurpool is None else (1,1), **kwargs)
    
    def __call__(self, x):
        x = self.conv2d(x)
        if self.blurpool is not None: x = self.blurpool(x)
        return x

class LPFMaxPooling2D():
    def __init__(self, pool_size,
        pad_type='reflect', lpf_size=3, strides=(1,1),
        channels=None, pad_off=0, **kwargs):
        if isinstance(strides, int):
            strides=(strides,strides)
        
        if strides[0] != strides[1]:
            raise 
        if strides[0] > MAX_SUPPORTED_KERNEL_SIZE:
            raise 
        
        if 'name' in kwargs:
            blurpool_name = kwargs['name']+'_blurpool'
        else:
            blurpool_name = None
        
        if strides[0] > 1 and lpf_size>0:
            self.blurpool = BlurPool(pad_type=pad_type, filt_size=lpf_size, 
                            stride=strides[0], channels=channels, 
                            pad_off=pad_off, name=blurpool_name)
        else:
            self.blurpool=None
        self.max = MaxPooling2D(pool_size, strides=strides if self.blurpool is None else (1,1), **kwargs)
    
    def __call__(self, x):
        x = self.max(x)
        if self.blurpool is not None: x = self.blurpool(x)
        return x

def LPFAveragePooling2D(pool_size, strides, lpf_size, **kwargs):
    if isinstance(strides, int):
        strides=(strides,strides)
    return BlurPool(filt_size=max(pool_size, lpf_size), stride=strides[0], **kwargs)