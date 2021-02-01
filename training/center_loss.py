import keras
import tensorflow as tf
import keras.backend as K

def _center_loss_impl(features, label, alfa, nrof_classes, centers, features_dim):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    #assert nrof_features == features_dim
    label = K.argmax(label, axis=1)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss


def center_loss(features, additional_loss, alfa, nrof_classes, center_loss_weight, features_dim):
    centers = K.zeros([nrof_classes, features_dim])
    def loss(y_true, y_pred):
        #print(y_true.shape)
        return additional_loss(y_true, y_pred) + center_loss_weight*_center_loss_impl(features, y_true, alfa, nrof_classes, centers, features_dim)
    return loss
