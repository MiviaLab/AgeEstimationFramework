'''
Models provided with relu limited between zero and 100. For regression purpose, age recognition task.
'''

import keras
import sys
import numpy as np
import tensorflow as tf

# # previously used
# def mobilenet_224_build_custom(input_shape, num_classes, weights):
#     m1 = keras.applications.mobilenet_v2.MobileNetV2(input_shape, 1.0, include_top=False, weights=weights)
#     features = keras.layers.GlobalAveragePooling2D()(m1.output)
#     assert num_classes == 1, "num_classes not 1"
#     x = keras.layers.Dense(num_classes, use_bias=True, activation=age_relu, name="regression_layer")(features)
#     return keras.Model(m1.input, x), features


def age_relu(x):
    return keras.backend.relu(x, max_value=100)

def Hswish(x):
        return x * tf.nn.relu6(x + 3) / 6

def HSigmoid(x):
    return tf.nn.relu6(x + 3) / 6

def mobilenet_v3_small_build(input_shape=(224, 224, 3), num_classes=1, weights="imagenet"):
    print("Building mobilenet small v3", input_shape, "- num_classes", num_classes, "- weights", weights)
    sys.path.append("MobileNetV3")
    from MobileNetV3.MobileNetV3 import MobileNetV3_small
    # input_tensor = keras.layers.Lambda(lambda t: t / 127.5)(keras.layers.Input(shape=input_shape))
    # m1 = MobileNetV3_small(input_shape=input_shape, input_tensor=input_tensor, include_top=True, weights=weights)
    m1 = MobileNetV3_small(input_shape=input_shape, include_top=True, weights=weights)
    features = m1.layers[-4].output
    assert num_classes == 1, "num_classes not 1"
    
    x = keras.layers.Conv2D(num_classes,(1,1),strides=(1,1),padding='same',use_bias=True,name='logits')(features)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Activation(age_relu)(x)

    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features

def mobilenet_v3_large_build(input_shape=(224, 224, 3), num_classes=1, weights="imagenet"):
    print("Building mobilenet large v3", input_shape, "- num_classes", num_classes, "- weights", weights)
    sys.path.append("MobileNetV3")
    from MobileNetV3.MobileNetV3 import MobileNetV3_large
    
    # input_tensor = keras.layers.Lambda(lambda t: t / 127.5)(keras.layers.Input(shape=input_shape))
    # m1 = MobileNetV3_large(input_shape=input_shape, input_tensor=input_tensor, include_top=True, weights=weights)

    m1 = MobileNetV3_large(input_shape=input_shape, include_top=True, weights=weights)
    features = m1.layers[-4].output
    assert num_classes == 1, "num_classes not 1"
    
    x = keras.layers.Conv2D(num_classes,(1,1),strides=(1,1),padding='same',use_bias=True,name='logits')(features)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Activation(age_relu)(x)

    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features

def resnet50_build(input_shape=(224, 224, 3), num_classes=2, weights="imagenet"):
    print("Building resnet50", input_shape, "- num_classes", num_classes, "- weights", weights)
    m1 = keras.applications.ResNet50V2(input_shape=input_shape, weights=weights, include_top=False, pooling="avg")
    features = m1.layers[-1].output
    if num_classes == 1:
        x = keras.layers.Dense(1, use_bias=True, activation=age_relu, name="regression_layer")(features)
    else:
        x = keras.layers.Dense(num_classes, use_bias=True, activation="softmax", name="Logits")(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features

def senet_model_build(input_shape=(224, 224, 3), num_classes=1, weights="imagenet"):
    print("Building senet", input_shape, "- num_classes", num_classes, "- weights", weights)
    sys.path.append('keras-squeeze-excite-network')
    from keras_squeeze_excite_network.se_resnet import SEResNet
    m1 = SEResNet(weights=weights, input_shape=input_shape, include_top=True, pooling='avg', weight_decay=0)  # , lpf_size=args.lpf_size)
    features = m1.layers[-2].output
    # assert num_classes == 1, "num_classes not 1"
    if num_classes == 1:
        x = keras.layers.Dense(1, use_bias=True, activation=age_relu, name="regression_layer")(features)
    else:
        x = keras.layers.Dense(num_classes, use_bias=True, activation="softmax", name="Logits")(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def mobilenet_224_build(input_shape=(224, 224, 3), num_classes=1, weights="imagenet"):
    print("Building mobilenet v2", input_shape, "- num_classes", num_classes, "- weights", weights)
    m1 = keras.applications.mobilenet_v2.MobileNetV2(input_shape, 1.0, include_top=True, weights=weights)
    features = m1.layers[-2].output
    assert num_classes == 1, "num_classes not 1"
    x = keras.layers.Dense(num_classes, use_bias=True, activation=age_relu, name="regression_layer")(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def mobilenet_96_build(input_shape=(96,96,3), num_classes=1, weights="imagenet"):
    print("Building mobilenet v2 96 0.75", input_shape, "- num_classes", num_classes, "- weights", weights)
    m1 = keras.applications.mobilenet_v2.MobileNetV2(input_shape, 0.75, include_top=True, weights=weights)
    features = m1.layers[-2].output
    assert num_classes == 1, "num_classes not 1"
    x = keras.layers.Dense(num_classes, use_bias=True, activation=age_relu, name="regression_layer")(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features
    

def densenet_121_build(input_shape=(224, 224, 3), num_classes=1, weights="imagenet", lpf_size=1):
    print("Building densenet121bc", input_shape, "- num_classes", num_classes, "- weights", weights)
    sys.path.append('keras_vggface')
    from keras_vggface.densenet import DenseNet121
    m1 = DenseNet121(include_top=True, input_shape=input_shape, weights=weights, pooling='avg', lpf_size=lpf_size)
    features = m1.layers[-2].output
    assert num_classes == 1, "num_classes not 1"
    x = keras.layers.Dense(num_classes, use_bias=True, activation=age_relu, name="regression_layer")(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def vgg16_keras_build(input_shape=(224, 224, 3), num_classes=1, weights="imagenet"):
    print("Building vgg16", input_shape, "- num_classes", num_classes, "- weights", weights)
    from keras.applications.vgg16 import VGG16

    # method 1: 1440 
    input_tensor = keras.layers.Input(shape=input_shape)
    input_tensor = keras.layers.Lambda(keras.applications.vgg16.preprocess_input, arguments={'mode': 'tf', 'data_format':'channels_last'})(input_tensor)

    # method 2: 4000
    # input_tensor = keras.layers.Input(shape=input_shape)
    # input_tensor = keras.layers.Lambda(lambda t: (t / 127.5) - 1.)(input_tensor)

    # method 3: 1400
    # input_tensor = keras.layers.Input(shape=input_shape)
    # input_tensor = keras.layers.Lambda(lambda t: t / np.array([163.5047, 151.1173, 131.0912]))(input_tensor)

    # method 4: 4000
    # input_tensor = keras.layers.Input(shape=input_shape)
    # input_tensor = keras.layers.Lambda(keras.applications.vgg16.preprocess_input, arguments={'mode': 'caffe'})(input_tensor)

    # method 5:
    # input_tensor = keras.layers.Input(shape=input_shape)
    # input_tensor = keras.applications.vgg16.preprocess_input(input_tensor, mode="tf", data_format='channels_last')

    # method 6 in training: 
    # input_tensor = keras.layers.Input(shape=input_shape)
    # input_tensor = keras.layers.Lambda(lambda t: t / 127.5)(input_tensor)
    m1 = VGG16(include_top=True, weights=weights, input_tensor=input_tensor)
    features = m1.layers[-2].output
    assert num_classes == 1, "num_classes not 1"
    x = keras.layers.Dense(num_classes, use_bias=True, activation=age_relu, name="regression_layer")(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


    # method 7:
    # def custom_preprocessing(img):
    #     upper = np.array([163.5047, 151.1173, 123.9088])
    #     lower = np.array([ -91.4953, -103.8827, -131.0912])

    #     maximo = np.max(img)
    #     minimo = np.min(img)

    #     abso_max = np.max(np.abs(img))

    #     return img/abso_max


    # input_tensor = keras.layers.Input(shape=input_shape)
    # input_tensor = keras.layers.Lambda(lambda t: custom_preprocessing(t))(input_tensor)

    # method screen
    # input_tensor = keras.layers.Input(shape=input_shape)
    # input_tensor = keras.layers.Lambda(lambda t: t / 127.5)(input_tensor) # np.max(np.abs(t))
    # m1 = VGG16(include_top=True, weights=weights, input_tensor=input_tensor)
    # m1.layers[-1].activation = keras.activations.relu
    # features = m1.layers[-1].output
    # print("@@@@@@@@@@@@@@@", m1.layers[-1].activation)
    # assert num_classes == 1, "num_classes not 1" 
    # x = keras.layers.Dense(num_classes, use_bias=True, activation=age_relu, name="regression_layer")(features)
    # model = keras.models.Model(m1.input, x)
    # for l in model.layers: l.trainable = True
    # return model, features

    # method no_screen
    # m1 = VGG16(include_top=True, weights=weights, input_shape=input_shape)
    # features = m1.layers[-1].output
    # assert num_classes == 1, "num_classes not 1"
    # x = keras.layers.Dense(num_classes, use_bias=True, activation=age_relu, name="regression_layer")(features)
    # model = keras.models.Model(m1.input, x)
    # for l in model.layers: l.trainable = True
    # return model, features

    # backup
    # input_tensor = keras.layers.Input(shape=input_shape)
    # input_tensor = keras.layers.Lambda(lambda t: t / 127.5)(input_tensor) # np.max(np.abs(t))
    # m1 = VGG16(include_top=False, weights=weights, input_tensor=input_tensor, pooling='avg')
    # features = m1.layers[-1].output
    # assert num_classes == 1, "num_classes not 1"
    # # x = keras.layers.Flatten(name='flatten')(features)
    # x = keras.layers.Dense(512, activation='relu', name='fc1')(features)
    # x = keras.layers.Dense(256, activation='relu', name='fc2')(x)
    # x = keras.layers.Dense(num_classes, use_bias=True, activation=age_relu, name="regression_layer")(x)
    # model = keras.models.Model(m1.input, x)
    # for l in model.layers: l.trainable = True
    # return model, features

    # Original
    # m1 = VGG16(include_top=True, weights=weights, input_tensor=None, input_shape=input_shape)
    # features = m1.layers[-2].output
    # assert num_classes == 1, "num_classes not 1"
    # x = keras.layers.Dense(num_classes, use_bias=False, activation=age_relu, name="regression_layer")(features)
    # return keras.models.Model(m1.input, x), features




# def mobilenet_64_build(input_shape, num_classes=1):
#     from scratch_models.mobile_net_v2_keras import MobileBioNetv2
#     m1 = MobileBioNetv2(input_shape=input_shape, width_multiplier=0.5)
#     features = m1.layers[-2].output
#     assert num_classes == 1, "num_classes not 1"
#     x = keras.layers.Dense(num_classes, use_bias=True, activation=age_relu, name="regression_layer")(features)
#     return keras.Model(m1.input, x), features





# def xception_build(input_shape=(299,299,3), num_classes=1, weights="imagenet", lpf_size=1):
#     sys.path.append('keras_vggface')
#     from keras_vggface.xception import Xception
#     m1 = Xception(input_shape=input_shape, weights=weights, include_top=True, pooling='avg', lpf_size=lpf_size)
#     features = m1.layers[-2].output
#     assert num_classes == 1, "num_classes not 1"
#     x = keras.layers.Dense(num_classes, use_bias=True, activation=age_relu, name="regression_layer")(features)
#     return keras.Model(m1.input, x), features


# def squeezenet_build(input_shape=(224, 224, 3), num_classes=1, weights="imagenet"):
#     # TODO test
#     sys.path.append('keras-squeezenet')
#     from keras_squeezenet import SqueezeNet
#     model = SqueezeNet(include_top=False, weights=weights) # weights = 'imagenet'
#     features = model.output
#     assert num_classes == 1, "num_classes not 1"
#     x = keras.layers.Dropout(0.5, name='drop9')(features)
#     x = keras.layers.Convolution2D(num_classes, (1, 1), padding='valid', name='conv10')(x)
#     x = keras.layers.Activation('relu', name='relu_conv10')(x)
#     x = keras.layers.GlobalAveragePooling2D()(x)
#     x = keras.layers.Dense(num_classes, use_bias=True, activation=age_relu, name="regression_layer")(features)
#     return keras.models.Model(model.input, x), features

# def shufflenet_224_build(input_shape, num_classes, weights):
#     # TODO test
#     sys.path.append('keras-shufflenetV2')
#     from shufflenetv2 import ShuffleNetV2
#     m1 = ShuffleNetV2(input_shape=input_shape, classes=num_classes, include_top=True, scale_factor=1.0, weights=weights)
#     features = m1.layers[-2].output
#     assert num_classes == 1, "num_classes not 1"
#     x = keras.layers.Dense(num_classes, use_bias=True, activation=age_relu, name="regression_layer")(features)
#     return keras.Model(m1.input, x), features


# def vggface_custom_build(input_shape, num_classes, weights, net, lpf_size):
#     # TODO test
#     sys.path.append('keras_vggface')
#     from keras_vggface.vggface import VGGFace
#     return VGGFace(model=net, weights=weights, input_shape=input_shape, classes=num_classes, lpf_size=lpf_size)


# # def vgg16_keras_build_old(input_shape, num_classes, weights):
# #     from keras.applications.vgg16 import VGG16, preprocess_input
# #     m1 = VGG16(include_top=True, weights=weights, input_tensor=None, input_shape=input_shape, pooling=None)

# #     extra_input_tensor = keras.layers.Input(shape=input_shape)
# #     # prep_lambda = keras.layers.Lambda(preprocess_input, arguments={'mode': 'torch'})
# #     # prep_lambda = keras.layers.Lambda(preprocess_input, arguments={'mode': 'tf'})
# #     # prep_lambda = keras.layers.Lambda(lambda t: t / 128)
# #     prep_lambda = keras.layers.Lambda(lambda t: (t / 127.5) - 1.)
# #     lambda_preprocess = prep_lambda(extra_input_tensor)
# #     # m1.layers.pop(0)
# #     features = keras.models.Model(m1.input, m1.layers[-2].output)(lambda_preprocess)
# #     x = keras.layers.Dense(num_classes, activation='softmax', name='Logits')(features)
# #     model = keras.models.Model(extra_input_tensor, x)

# #     # ORIGINAL STRUCTURE
# #     # features = m1.layers[-2].output
# #     # x = keras.layers.Dense(NUM_CLASSES, activation='softmax', name='Logits')(features)
# #     # model = keras.models.Model(m1.input, x)

# #     return model, features

# # def regression_layer(activation="linear"):
# #     return keras.layers.Dense(1, activation=activation, use_bias=True, name='regression')

# # def age_regression_custom_layer(bottomlayer):
# #     dense = keras.layers.Dense(1, use_bias=True, name='regression')(bottomlayer)
# #     return keras.layers.Activation('relu', max_value=100)(dense)

    



