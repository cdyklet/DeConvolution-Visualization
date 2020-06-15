import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import cv2
import h5py
from imageio import imwrite
from numba import cuda
from six.moves.urllib.request import urlretrieve

gpu = tf.config.experimental.get_visible_devices("GPU")[0]
tf.config.experimental.set_memory_growth(gpu, enable=True)
layers_name = [
    "block1_conv1",
    "block1_conv2",
    "block2_conv1",
    "block2_conv2",
    "block3_conv1",
    "block3_conv2",
    "block3_conv3",
    "block4_conv1",
    "block4_conv2",
    "block4_conv3",
    "block5_conv1",
    "block5_conv2",
    "block5_conv3",
]
layers_name = layers_name[::-1]


class UpMaxPooling(tf.keras.layers.Layer):
    def __init__(self, strides=(2, 2), ksize=(2, 2)):
        super(UpMaxPooling, self).__init__()
        self.ksize = ksize
        self.strides = strides

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape
        return mask_shape[1] * self.ksize[0], mask_shape[2] * self.ksize[1], mask_shape[3]

    def call(self, inputs):
        input, ind = inputs[0], inputs[1]
        y, x, f = self.compute_output_shape(input.shape)
        pool = tf.reshape(input, (-1, 1))
        ind = tf.cast(tf.reshape(ind, (-1, 1)), tf.int32)
        input = tf.keras.layers.Concatenate(axis=1)([input for _ in range(self.ksize[0])])
        input = tf.keras.layers.Concatenate(axis=2)([input for _ in range(self.ksize[1])])
        input = tf.reshape(input, (-1, 1))
        ret = tf.scatter_nd(ind, pool, tf.shape(input))
        ret = tf.reshape(ret, shape=[-1, y, x, f])
        return ret


class DeConv(tf.keras.layers.Layer):
    def __init__(self, filter_num, strides=(1, 1), kernel_size=(3, 3), padding="same", activation="relu"):
        super(DeConv, self).__init__()
        self.filter = filter_num
        self.strides = strides
        self.padding = padding.upper()
        self.kernel_size = kernel_size
        self.activation = activation

    def build(self, input_shape):
        #input_shape = input_shape
        input_channel = input_shape[-1]
        input_width = input_shape[2]
        if len(self.kernel_size) == 1:
            kernel_size = (self.kernel_size[0], self.kernel_size[0])
        else:
            kernel_size = tuple(self.kernel_size)
        f = self.kernel_size[0]
        s = self.strides[0]
        if self.padding == "VALID":
            h = (input_width - 1) * s + f
        else:
            h = input_width * s
        self.h = h
        # self._output_shape = (input_num, h, h, self.filter)
        kernel_size = kernel_size + (self.filter, input_channel)
        self.kernel = self.add_weight(name="kernel", shape=kernel_size, dtype=tf.float32)
        self.built = True

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        _output_shape = (batch_size, self.h, self.h, self.filter)
        y = tf.nn.conv2d_transpose(x, self.kernel, output_shape=_output_shape, strides=self.strides, padding=self.padding)
        if self.activation:
            y = tf.nn.relu(y)
        return y


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


def vgg16_deconv(load_weight=True):
    conv_input = tf.keras.Input(shape=(7, 7, 512), dtype=tf.float32)
    arg5_input = tf.keras.Input(shape=(7, 7, 512), dtype=tf.int64)
    arg4_input = tf.keras.Input(shape=(14, 14, 512), dtype=tf.int64)
    arg3_input = tf.keras.Input(shape=(28, 28, 256), dtype=tf.int64)
    arg2_input = tf.keras.Input(shape=(56, 56, 128), dtype=tf.int64)
    arg1_input = tf.keras.Input(shape=(112, 112, 64), dtype=tf.int64)
    # block 5
    conv = UpMaxPooling()([conv_input, arg5_input])
    conv = DeConv(512)(conv)
    conv = DeConv(512)(conv)
    conv = DeConv(512)(conv)
    # block 4
    conv = UpMaxPooling()([conv, arg4_input])
    conv = DeConv(512)(conv)
    conv = DeConv(512)(conv)
    conv = DeConv(256)(conv)
    # block 3
    conv = UpMaxPooling()([conv, arg3_input])
    conv = DeConv(256)(conv)
    conv = DeConv(256)(conv)
    conv = DeConv(128)(conv)
    # block 2
    conv = UpMaxPooling()([conv, arg2_input])
    conv = DeConv(128)(conv)
    conv = DeConv(64)(conv)
    # block 1
    conv = UpMaxPooling()([conv, arg1_input])
    conv = DeConv(64)(conv)
    conv = DeConv(3)(conv)
    arg_input = [arg5_input, arg4_input, arg3_input, arg2_input, arg1_input]
    model = tf.keras.Model(inputs=[conv_input, arg_input], outputs=conv)
    if load_weight:
        # first check local weight file is exist or not, then load weight using h5py
        top_file_path = "./weight/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
        if not os.path.exists(top_file_path):
            top_file_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
            download_weight(top_file_url, top_file_path)
        # For notop situation, path and url are "./weight/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5" and 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5' respectively.
        weights = []
        with h5py.File(top_file_path, "r") as f:
            for name in layers_name:
                weights.append(f[name][name + "_W_1:0"][()])
        model.set_weights(weights)
    return model


def deconv_idx(i=1):
    i -= 1
    deconv_layer_idx = [3, 4, 5, 8, 9, 10, 13, 14, 15, 18, 19, 22, 23]
    return deconv_layer_idx[::-1][i]


def backward(conv, argmax_pool, layer=1):
    model = vgg16_deconv()
    _output = model.output
    layer = deconv_idx(layer)
    _conv_input = model.layers[layer].input
    _arg_input = []
    _arg_input_idx = [1, 6, 11, 16, 20]
    for i in _arg_input_idx[::-1]:
        if layer > i:
            break
        _arg_input.append(model.layers[i].input)
    assert len(argmax_pool) == len(_arg_input)
    if _arg_input:
        _inputs = [_conv_input, *_arg_input]
        inputs = [conv, *argmax_pool]
    else:
        _inputs = _conv_input
        inputs = conv
    fun = tf.keras.backend.function(inputs=_inputs, outputs=_output)
    outputs = fun(inputs)
    return outputs


def load_data_from_h5file(file_dir):
    with h5py.File(file_dir, "r") as f:
        conv = tf.Variable(f["conv"][()], dtype=tf.float32)
        arg = []
        keys = f["argmax_pool_indices"].keys()
        for key in keys:
            value = f["argmax_pool_indices"][key][()]
            arg.append(tf.Variable(value, dtype=tf.int64))
    return conv, arg


if __name__ == "__main__":
    # # print(model.summary())
    test_file_dir = "./output/trial.h5"
    conv, arg = load_data_from_h5file(test_file_dir)
    print(conv.shape)
    print(len(arg))
    image = backward(conv, arg, layer=13)
    print(image.shape)

    # deconv = model(conv, ind, i=2)
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # deconv = (deconv.numpy()[0] + mean) * std * 255
    # deconv = deconv.astype(np.int8)
    # deconv = np.clip(deconv, 0, 255)
    # image_path = "./test.png"
    # cv2.imwrite(image_path, deconv)
