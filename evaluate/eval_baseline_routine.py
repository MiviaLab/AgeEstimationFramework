import os
import sys
import keras
import time
import numpy as np
from tqdm import tqdm

############## PARAMS ################ #TODO argparse
GPU = 1
modelpath = "/user/gdiprisco/age/trained/_netresnet50_datasetvggface2_age_pretrainingimagenet_preprocessingvggface2_augmentationdefault_batch128_lr0.005_0.2_20_sel_gpu1,2_ngpus2_training-epochs70_20210117_112451/checkpoint.43.hdf5"

partition = "test"
input_shape = (224,224,3)#(112,112,3)
######################################

sys.path.append("../training")
from model_build import age_relu
custom_objects = {"age_relu" : age_relu}

sys.path.append("../../multitask/evaluate")
from memory_usage import keras_model_memory_usage_in_bytes

sys.path.append("../dataset")
from lfw_dataset_age import LFWPlusDatasetAge as Dataset
# from vgg2_dataset_age import Vgg2DatasetAge as Dataset

# def mobilenet_v3_large_build(input_shape, num_classes, weights=None):
#     sys.path.append("../../multitask/training")
#     from MobileNetV3.MobileNetV3 import MobileNetV3_large
#     print("Building mobilenet large v3", input_shape, "- num_classes", num_classes, "- weights", weights)
#     m1 = MobileNetV3_large(input_shape=input_shape, include_top=True, weights=weights)
#     features = m1.layers[-4].output
#     x = keras.layers.Conv2D(num_classes, (1, 1), strides=(1, 1), padding='same', use_bias=True, name='logits')(features)
#     x = keras.layers.Flatten()(x)
#     x = keras.layers.Activation('softmax')(x)
#     model = keras.models.Model(m1.input, x)
#     for l in model.layers: l.trainable = True
#     return model

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    print("Loading model on GPU", os.environ["CUDA_VISIBLE_DEVICES"], "...")
    # ########### MobileNet v3 ###########
    # model = mobilenet_v3_large_build(input_shape=input_shape, num_classes=2)
    # model.compile(loss=keras.losses.categorical_crossentropy, optimizer='sgd', metrics=[keras.metrics.categorical_accuracy])
    # model.load_weights(modelpath)

    # ########### SEResNet50 ###########
    # model = keras.models.load_model(modelpath)

    ########### ResNet50 multi-gpu extraction lambda ###########
    model = keras.models.load_model(modelpath, custom_objects=custom_objects)
    lambda_model = model.layers[-2]
    lambda_model.summary()
    model = keras.models.Model(lambda_model.layers[0].output, lambda_model.layers[-1].output)
    model.compile(
        optimizer=keras.optimizers.sgd(momentum=0.9),
        loss=keras.losses.mean_squared_error,
        metrics=[keras.metrics.mean_absolute_error]
    )
    model.summary()

    print("Model loaded.")
    print("Loading dataset...")
    dataset = Dataset(
        partition=partition,
        target_shape=input_shape,
        augment=False,
        preprocessing='vggface2')
    #     imagesdir='/user/gdiprisco/gender_refactored/dataset/data/gender-access/lfw_cropped',
    #     csvmeta='/user/gdiprisco/gender_refactored/dataset/data/gender-access/lfw_theirs_<gender>.csv',
    #     cache_dir='/user/gdiprisco/gender_refactored/dataset/cache/'
    # )
    print("Dataset loaded.")
    print("Evaluating model...")
    batch_size = 16
    data_gen = dataset.get_generator(batch_size)
    score = model.evaluate(data_gen, verbose=1, workers=4)
    score = {out: score[i] for i, out in enumerate(model.metrics_names)}
    print("Model evaluated.")
    print(" ------ ACCURACY ------")
    print(score)
    print(" ----------------------")

    # print("\nStart inference time test: batch size 1 ...")
    # data_gen = dataset.get_generator(batch_size=1)
    # start_time = time.time()
    # for batch in tqdm(data_gen):
    #     _ = model.predict(batch[0])
    # spent_time = time.time() - start_time
    # batch_average_time = spent_time / len(data_gen)
    # fps = 1/batch_average_time
    # GPU_bytes = keras_model_memory_usage_in_bytes(model=model, batch_size=1)
    # print(" --- INFERENCE TEST RUNNED ---")
    # print("Evaluate time %d s" % spent_time)
    # print("LATENCY: %.10f s, FPS: %.3f" % (batch_average_time, fps))
    # print("PARAMETERS:", model.count_params())
    # print("Memory usage {} bytes".format(GPU_bytes))
    # print(" -----------------------------")
    