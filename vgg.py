import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from six.moves.urllib.request import urlretrieve
import cv2
import numpy as np
import h5py
import multiprocessing
from numba import cuda


gpu = tf.config.experimental.get_visible_devices("GPU")[0]
tf.config.experimental.set_memory_growth(gpu, enable=True)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
layers_name = [
    "block1_conv1",
    "block1_conv2",
    "block1_pool",
    "block2_conv1",
    "block2_conv2",
    "block2_pool",
    "block3_conv1",
    "block3_conv2",
    "block3_conv3",
    "block3_pool",
    "block4_conv1",
    "block4_conv2",
    "block4_conv3",
    "block4_pool",
    "block5_conv1",
    "block5_conv2",
    "block5_conv3",
    "block5_pool",
    "fc1",
    "fc2",
    "flatten",
    "predictions",
]

layers_weight_indices = [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 21]
top_file_path = "./weight/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
notop_file_path = "./weight/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"


class Pool_argmax(tf.keras.layers.Layer):
    def __init__(self, ksize=(2, 2), strides=(2, 2), padding="SAME", name=None):
        super(Pool_argmax, self).__init__()
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        if isinstance(padding, str):
            padding = padding.capitalize()
        self._name_pre = name

    def call(self, x):
        return tf.nn.max_pool_with_argmax(
            x, ksize=self.ksize, strides=self.strides, padding=self.padding, name=self._name_pre, include_batch_in_index=True
        )


def download_weight(url, filename):
    error_msg = "fetch failure {}: {} --{}"
    try:
        try:
            urlretrieve(url, filename)
        except HTTPError as e:
            raise Exception(error_msg.format(url, e.code, e.msg))
        except URLError as e:
            raise Exception(error_msg.format(url, e.errno, e.reason))
    except (KeyboardInterrupt, Exception) as e:
        if os.path.exist(filename):
            os.path.remove(filename)
        raise
    return


def vgg16(load_weight=True):
    input = tf.keras.Input(shape=(224, 224, 3))
    conv_1x1 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), name="conv_1x1", activation="relu", padding="same")(input)
    conv_1x2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), name="conv_1x2", activation="relu", padding="same")(conv_1x1)
    pool1, arg_1 = Pool_argmax(name="pool1")(conv_1x2)
    # block2
    conv_2x1 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), name="conv_2x1", activation="relu", padding="same")(pool1)
    conv_2x2 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), name="conv_2x2", activation="relu", padding="same")(conv_2x1)
    pool2, arg_2 = Pool_argmax(name="pool2")(conv_2x2)
    # block 3
    conv_3x1 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), name="conv_3x1", activation="relu", padding="same")(pool2)
    conv_3x2 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), name="conv_3x2", activation="relu", padding="same")(conv_3x1)
    conv_3x3 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), name="conv_3x3", activation="relu", padding="same")(conv_3x2)
    pool3, arg_3 = Pool_argmax(name="pool3")(conv_3x3)
    # block 4
    conv_4x1 = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), name="conv_4x1", activation="relu", padding="same")(pool3)
    conv_4x2 = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), name="conv_4x2", activation="relu", padding="same")(conv_4x1)
    conv_4x3 = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), name="conv_4x3", activation="relu", padding="same")(conv_4x2)
    pool4, arg_4 = Pool_argmax(name="pool4")(conv_4x3)
    # block 5
    conv_5x1 = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), name="conv_5x1", activation="relu", padding="same")(pool4)
    conv_5x2 = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), name="conv_5x2", activation="relu", padding="same")(conv_5x1)
    conv_5x3 = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), name="conv_5x3", activation="relu", padding="same")(conv_5x2)
    pool5, arg_5 = Pool_argmax(name="pool5")(conv_5x3)
    dense = tf.keras.layers.Flatten()(pool5)
    dense1 = tf.keras.layers.Dense(4096, activation="relu")(dense)
    drop1 = tf.keras.layers.Dropout(0.5)(dense1)
    dense2 = tf.keras.layers.Dense(4096, activation="relu")(drop1)
    drop2 = tf.keras.layers.Dropout(0.5)(dense2)
    soft1 = tf.keras.layers.Dense(1000, activation="softmax")(drop2)
    val = [soft1, [arg_1, arg_2, arg_3, arg_4, arg_5]]
    model = tf.keras.Model(inputs=input, outputs=val)
    if load_weight:
        # first check local weight file is exist or not, then load weight using h5py
        top_file_path = "./weight/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
        if not os.path.exists(top_file_path):
            top_file_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
            download_weight(top_file_url, top_file_path)
        # For notop situation, path and url are "./weight/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5" and 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5' respectively.
        with h5py.File(top_file_path, "r") as f:
            ret = []
            for name in layers_name:
                layer = f[name]
                ret.extend([layer[k][()] for k in layer.keys()])
        model.set_weights(ret)
    return model


def conv_idx(i=1):
    i -= 1
    conv_layer_idx = [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17]
    return conv_layer_idx[i]


def conv_filter(i=1):
    i -= 1
    conv_filters = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    return conv_filters[i]


def forward(image, layer=13, load_weight=True):
    # cuda.start()
    model = vgg16(load_weight)
    input = model.input
    layer = conv_idx(layer)
    arg_idx = [3, 6, 10, 14, 18]
    arg_pos = []
    for idx in arg_idx:
        if idx > layer:
            break
        arg_pos.append(model.layers[idx].output[1])
    if layer in arg_idx:
        output = model.layers[layer].output[0]
    else:
        output = model.layers[layer].output
    outputs = [output, arg_pos]
    fun = tf.keras.backend.function(inputs=input, outputs=outputs)
    y_pred = fun(image)
    # device = cuda.get_current_device()
    # device.reset()
    return y_pred


def write_data_h5file(test_data_dir, y_pred):
    with h5py.File(test_data_dir, "w") as f:
        f.create_dataset("conv", dtype=np.float32, shape=y_pred[0].shape, data=y_pred[0])
        arg_pool = f.create_group("argmax_pool_indices")
        for i, ind_max in enumerate(y_pred[1]):
            arg_pool.create_dataset(name="arg_" + str(i + 1), data=y_pred[1][i], dtype=np.int64)
    return


if __name__ == "__main__":
    image_file = "./data/dog.jpg"
    image = cv2.imread(image_file)
    image = cv2.resize(image, (224, 224))
    image = image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = tf.constant(image, dtype=tf.float32)
    image = image[tf.newaxis, ...]
    y_pred = forward(image, layer=13)
    print(y_pred[0].shape)
    print("finished")
    test_data_dir = "./output/trial.h5"
    write_data_h5file(test_data_dir, y_pred)
