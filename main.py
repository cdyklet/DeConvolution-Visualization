import os
os.environ['MIN_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
from utils import decode_predictions
import tensorflow as tf
import numpy as np
import cv2
from functools import partial
from vgg import forward, conv_filter
from vgg_deconv import backward


def load_images(img_path):
    # imread from img_path
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224)).astype(np.float32)
    # img = img /255.
    img = img[..., ::-1]
    # must normalize the pic by
    # mean = [0.455, 0.456, 0.406]
    # stf = [0.229, 0.224, 0.225]
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # img = (img - mean) / std
    mean = np.array([123.68, 116.78, 103.94])
    img = img - mean
    img = tf.Variable(img, dtype=tf.float32)
    img = img[tf.newaxis, ...]
    return img

#@tf.function
def vis_layer(conv, deconv, layer, image):
    """ 
    visualing the layer deconv layer
    """
    num_feat = conv_filter(layer)
    act_list = []
    new, arg = conv(image, layer=layer)
    new_feat_map = tf.Variable(new, dtype = tf.float32)
    for i in range(num_feat):
        choose_map = new_feat_map[0, :, :, i]
        activation = tf.reduce_max(choose_map)
        act_list.append(activation)
    act_list = np.array(act_list)
    mask = np.argmax(act_list)
    choose_map = new_feat_map[0, :, :, mask]
    max_activation = tf.reduce_max(choose_map)
    print("max_activation is {}".format(max_activation))
    # make zeros for other feature maps
    if mask == 0:
        new_feat_map[..., mask + 1 :].assign(tf.zeros_like(new_feat_map[..., mask + 1 :]))
    else:
        new_feat_map[..., :mask].assign(tf.zeros_like(new_feat_map[..., :mask]))
        if mask != num_feat - 1:
            new_feat_map[..., mask + 1 :].assign(tf.zeros_like(new_feat_map[..., mask + 1 :]))
    choose_map = tf.where(choose_map == max_activation, choose_map, tf.zeros_like(choose_map))
    new_feat_map[..., mask].assign(tf.expand_dims(choose_map,axis = 0))
    new_img = deconv(new_feat_map, arg, layer)
    new_img = new_img[0]
    # new_img = new_img[..., ::-1]
    new_max, new_min = np.max(new_img), np.min(new_img)
    new_img = (new_img - new_min) / (new_max - new_min) * 255
    new_img = new_img.astype(np.int16)
    return new_img, max_activation

if __name__ == '__main__':
    img_path = './data/cat.jpg'
    img = load_images(img_path)
    conv = forward
    deconv = backward
    img, activation = vis_layer(conv, deconv, layer = 12, image = img)
    print(np.max(img), np.min(img), img.dtype)
    plt.imshow(img)
    plt.show()
    print(img.shape)
    