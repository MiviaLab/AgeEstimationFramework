import tensorflow as tf
from keras import backend as K
from keras import utils as keras_utils
from keras.models import Model, load_model
from keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Activation, Input, Add, Lambda
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Softmax, Flatten, Permute, AvgPool2D
# ** to update custom Activate functions
from keras.utils.generic_utils import get_custom_objects
import os

EXT_ROOT = os.path.dirname(os.path.abspath(__file__))

class MobileNetV3():

    # out_dim, kernel, strides, expansion_dim, bias, se, activation, num_layers, ratio
    LARGE = [[16,  (3, 3), (1, 1), 16,  False, False, 'RE',  0, 4],
            [24,  (3, 3), (2, 2), 64,  False, False, 'RE',  1, 4],
            [24,  (3, 3), (1, 1), 72,  False, False, 'RE',  2, 4],
            [40,  (5, 5), (2, 2), 72,  False, True,  'RE',  3, 3],
            [40,  (5, 5), (1, 1), 120, False, True,  'RE',  4, 3.75],
            [40,  (5, 5), (1, 1), 120, False, True,  'RE',  5, 3.75],
            [80,  (3, 3), (2, 2), 240, False, False, 'HS',  6, 4],
            [80,  (3, 3), (1, 1), 200, False, False, 'HS',  7, 4],
            [80,  (3, 3), (1, 1), 184, False, False, 'HS',  8, 4],
            [80,  (3, 3), (1, 1), 184, False, False, 'HS',  9, 4],
            [112, (3, 3), (1, 1), 480, False, True,  'HS', 10, 4],
            [112, (3, 3), (1, 1), 672, False, True,  'HS', 11, 4],
            [160, (5, 5), (2, 2), 672, False, True,  'HS', 12, 4],
            [160, (5, 5), (1, 1), 960, False, True,  'HS', 13, 4],
            [160, (5, 5), (1, 1), 960, False, True,  'HS', 14, 4]]

    SMALL = [[16,  (3, 3), (2, 2), 16,  False, True,  'RE', 0, 2],
            [24,  (3, 3), (2, 2), 72,  False, False, 'RE', 1, 4],
            [24,  (3, 3), (1, 1), 88,  False, False, 'RE', 2, 4],
            [40,  (5, 5), (2, 2), 96,  False, True,  'HS', 3, 4],
            [40,  (5, 5), (1, 1), 240, False, True,  'HS', 4, 3.75], 
            [40,  (5, 5), (1, 1), 240, False, True,  'HS', 5, 3.75],
            [48,  (5, 5), (1, 1), 120, False, True,  'HS', 6, 3.75],
            [48,  (5, 5), (1, 1), 144, False, True,  'HS', 7, 3.6],
            [96,  (5, 5), (2, 2), 288, False, True,  'HS', 8, 4],
            [96,  (5, 5), (1, 1), 576, False, True,  'HS', 9, 4],
            [96,  (5, 5), (1, 1), 576, False, True,  'HS', 10, 4]]

    def __init__(self):
        self.model_type = None
        self.nlay = -1
        self.max_exp_lay_num = -1
        # ** update custom Activate functions
        get_custom_objects().update({'custom_activation': Activation(self.Hswish)})
        get_custom_objects().update({'custom_sigmoid': Activation(self.HSigmoid)})

    """ Define layers block functions """
    def Hswish(self,x):
        return x * tf.nn.relu6(x + 3) / 6

    def HSigmoid(self,x):
        return tf.nn.relu6(x + 3) / 6

    def __conv2d_block(self, _inputs, filters, kernel, strides, is_use_bias=False, padding='same', activation='RE',model_type='large'):

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if self.nlay<0:
            basename = 'conv_%d' % (self.nlay+1)
        elif self.nlay > self.max_exp_lay_num:
            basename = 'conv_%d' % (self.nlay-self.max_exp_lay_num)
        else:
            basename = 'expanded_conv_%d_expand'%self.nlay

        x = Conv2D(filters, kernel, strides= strides, padding=padding,use_bias=is_use_bias, name=basename)(_inputs)
        x = BatchNormalization(momentum=0.9,axis=channel_axis,name=basename+'_batch_normalization')(x)

        if activation == 'RE':
            x = ReLU(name=basename+'_activation')(x)
        elif activation == 'HS':
            x = Activation(self.Hswish, name=basename+'_activation')(x)
        else:
            raise NotImplementedError

        return x

    def __depthwise_block(self, _inputs, kernel=(3, 3), strides=(1, 1), activation='RE', is_use_se=True, num_layers=0,ratio=4):

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = DepthwiseConv2D(kernel_size=kernel, strides=strides, depth_multiplier=1, padding='same',name='expanded_conv_%d_depthwise'%self.nlay,use_bias=False)(_inputs)
        x = BatchNormalization(axis=channel_axis,name='expanded_conv_%d_depthwise_batch_normalization'%self.nlay)(x)

        if activation == 'RE':
            x = ReLU( name='expanded_conv_%d_depthwise_activation'%self.nlay)(x)
        elif activation == 'HS':
            x = Activation(self.Hswish, name='expanded_conv_%d_depthwise_activation'%self.nlay)(x)
        else:
            raise NotImplementedError

        if is_use_se:
            x = self.__se_block(x,ratio=ratio)

        return x

    def __global_depthwise_block(self, _inputs):
        assert _inputs._keras_shape[1] == _inputs._keras_shape[2]
        kernel_size = _inputs._keras_shape[1]
        x = DepthwiseConv2D((kernel_size, kernel_size), strides=(1, 1), depth_multiplier=1, padding='same')(_inputs)
        return x

    # ratio = 4 DEFAULT
    def __se_block(self, _inputs, ratio=4, pooling_type='avg'):
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        filters = _inputs._keras_shape[channel_axis]
        se_shape = (1, 1, filters)
        
        if pooling_type == 'avg':
            se = GlobalAveragePooling2D()(_inputs)
        elif pooling_type == 'depthwise':
            se = self.__global_depthwise_block(_inputs)
        else:
            raise NotImplementedError
        
        se = Reshape(se_shape)(se)
        se = Conv2D(int(filters / ratio), (1,1), strides=(1,1), activation='relu', padding='same',use_bias=True, name='expanded_conv_%d_squeeze_excite_conv_0'%self.nlay)(se)
        se = Conv2D(filters, (1,1), strides=(1,1), activation=self.HSigmoid,padding='same',use_bias=True, name='expanded_conv_%d_squeeze_excite_conv_1'%self.nlay)(se)

        if K.image_data_format() == 'channels_first':
            se = Permute((3, 1, 2))(se)

        x = multiply([_inputs, se])
        return x

    def __bottleneck_block(self, _inputs, out_dim, kernel, strides, expansion_dim, is_use_bias=False, is_use_se=True, activation='RE', num_layers=0,ratio=4, *args):

        with tf.name_scope('bottleneck_block'):
            # ** to high dim 
            bottleneck_dim = expansion_dim

            channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

            input_shape = K.int_shape(_inputs)
            r = strides == (1,1) and input_shape[3] == out_dim

            # ** pointwise conv 
            if self.nlay > 0:
                x = self.__conv2d_block(_inputs, 
                                        bottleneck_dim,
                                        kernel=(1, 1),
                                        strides=(1, 1),
                                        is_use_bias=is_use_bias,
                                        activation=activation,
                                        model_type=self.model_type)
            else:
                x = _inputs

            # ** depthwise conv
            x = self.__depthwise_block(x,
                                        kernel=kernel, 
                                        strides=strides,
                                        is_use_se=is_use_se,
                                        activation=activation,
                                        num_layers=num_layers,
                                        ratio=ratio)

            # ** pointwise conv
            x = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same',name='expanded_conv_%d_project'%self.nlay,use_bias=False)(x)
            x = BatchNormalization(axis=channel_axis,name='expanded_conv_%d_project_batch_normalization'%self.nlay)(x)
            
            if r:
                x = Add()([x, _inputs])

            self.nlay += 1
        return x

    def build(self,input_shape=(224, 224, 3), num_classes=1000, model_type='large', pooling_type='avg', include_top=True, weights=None, input_tensor=None):
        
        self.model_type = model_type
        self.max_exp_lay_num = 14 if self.model_type is 'large' else 10

        # ** input layer
        # inputs = Input(shape=input_shape)

        if input_tensor is None:
            inputs = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                inputs = Input(tensor=input_tensor, shape=input_shape)
            else:
                inputs = input_tensor

        # ** feature extraction layers
        net = self.__conv2d_block(inputs, 
                                    16,
                                    kernel=(3, 3),
                                    strides=(2, 2),
                                    is_use_bias=False,
                                    padding='same',
                                    activation='HS',
                                    model_type=model_type) 
        self.nlay += 1
        
        if self.model_type == 'large':
            config_list = MobileNetV3.LARGE
        elif self.model_type == 'small':
            config_list = MobileNetV3.SMALL
        else:
            raise NotImplementedError
            
        for config in config_list:
            net = self.__bottleneck_block(net, *config)
        
        # ** final layers 
        net = self.__conv2d_block(net,
                                960 if model_type is 'large' else 576, 
                                kernel=(1, 1), 
                                strides=(1, 1),
                                is_use_bias=False,
                                padding='same',
                                activation='HS',
                                model_type=model_type)

        if pooling_type == 'avg':
            net = GlobalAveragePooling2D()(net)
        elif pooling_type == 'depthwise':
            net = self.__global_depthwise_block(net)
        else:
            raise NotImplementedError

        pooled_shape = (1, 1, net._keras_shape[-1])
        net = Reshape(pooled_shape)(net)
        net = Conv2D(1280 if model_type is 'large' else 1024, (1, 1), strides=(1, 1), padding='same', use_bias=True,name='conv_2')(net)
        net = Activation(self.Hswish,name='conv_2_activation')(net)
        
        if include_top:
            net = Conv2D(num_classes,(1,1),strides=(1,1),padding='same',use_bias=True,name='logits')(net)
            net = Flatten()(net)
            net = Softmax()(net)

        if input_tensor is not None:
            inputs = keras_utils.get_source_inputs(input_tensor)

        model = Model(inputs=inputs, outputs=net)

        if weights:
            model.load_weights(weights)

        return model

    def load(self, h5_path):
        return load_model(h5_path,custom_objects={'Hswish':Activation(self.Hswish), 'HSigmoid':Activation(self.HSigmoid)})


def MobileNetV3_small(input_shape=(224, 224, 3), input_tensor=None, num_classes=1001, include_top=True, pooling='avg', weights="imagenet"):
    if input_shape == (224, 224, 3) and num_classes == 1001 and include_top and weights == "imagenet":
        weights = os.path.join(EXT_ROOT, 'mobilenet_v3_small_1.0_224_weights.h5')
    return _MobileNetV3_core(input_shape=input_shape,
                                input_tensor=input_tensor,
                                num_classes=num_classes,
                                include_top=include_top,
                                pooling=pooling,
                                weights=weights,
                                model_type="small")


def MobileNetV3_large(input_shape=(224, 224, 3), input_tensor=None, num_classes=1001, include_top=True, pooling='avg', weights="imagenet"):
    # if input_shape == (224, 224, 3) and num_classes == 1001 and include_top and weights == "imagenet":
    if num_classes == 1001 and include_top and weights == "imagenet":
        weights = os.path.join(EXT_ROOT, 'mobilenet_v3_large_1.0_224_weights.h5')
    return _MobileNetV3_core(input_shape=input_shape,
                                input_tensor=input_tensor,
                                num_classes=num_classes,
                                include_top=include_top,
                                pooling=pooling,
                                weights=weights,
                                model_type="large")


def _MobileNetV3_core(input_shape=(224, 224, 3), input_tensor=None, num_classes=1001, include_top=True, pooling='avg', weights=None, model_type="large"):
    return MobileNetV3().build(input_shape=input_shape,
                                input_tensor = input_tensor,
                                num_classes=num_classes,
                                model_type=model_type,
                                pooling_type='avg',
                                include_top=True,
                                weights=weights)

def test1():
    model_types = ['large','small']

    X = cv2.imread('panda.jpg')
    X = X / 127.5 - 1
    X = cv2.resize(X,(224,224),cv2.INTER_CUBIC)
    X = np.asarray([X])
    
    for model_type in model_types:
        model = MobileNetV3().build(input_shape=(224,224,3),
                                    num_classes=1001, 
                                    model_type=model_type, 
                                    pooling_type='avg', 
                                    include_top=True,
                                    weights='mobilenet_v3_%s_1.0_224_weights.h5'%(model_type))

        #print(model.summary())

        output = model.predict(X)
        predicted_class = np.argmax(output)
        confidence = output[0][predicted_class]

        if model_type is 'large':
            print('[LARGE] TF_PREDICTED = ( 389 , 0.956092 ) , KERAS_PREDICTED = ({0} , {1})'.format(predicted_class,confidence))
        else:
            print('[SMALL] TF_PREDICTED = ( 389 , 0.9898086 ) , KERAS_PREDICTED = ({0} , {1})'.format(predicted_class,confidence))


def test2():
    print("Panda detecting...")
    X = cv2.imread('panda.jpg')
    # X = X / 127.5 - 1
    X = cv2.resize(X,(224,224),cv2.INTER_CUBIC)
    X = np.asarray([X])

    input_tensor = Input(shape=(224, 224,3))
    input_tensor = Lambda(lambda t: (t / 127.5) - 1.)(input_tensor)

    model = MobileNetV3_large(input_tensor=input_tensor)
    output = model.predict(X)
    predicted_class = np.argmax(output)
    confidence = output[0][predicted_class]
    print('[LARGE] TF_PREDICTED = ( 389 , 0.956092 ) , KERAS_PREDICTED = ({0} , {1})'.format(predicted_class,confidence))

    del model

    model = MobileNetV3_small(input_tensor=input_tensor)
    output = model.predict(X)
    predicted_class = np.argmax(output)
    confidence = output[0][predicted_class]
    print('[SMALL] TF_PREDICTED = ( 389 , 0.9898086 ) , KERAS_PREDICTED = ({0} , {1})'.format(predicted_class,confidence))



if __name__ == '__main__':

    from cv2 import cv2
    import numpy as np
    import os    
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    test1()

    # Test redefined

    test2()

        
