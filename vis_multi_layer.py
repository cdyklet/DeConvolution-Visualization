import matplotlib.pyplot as plt
from utils import decode_predictions
import tensorflow as tf
import numpy as np
import cv2
from vgg import forward, conv_filter
from vgg_deconv import backward
from main import load_images, vis_layer
import multiprocessing as mp

gpu = tf.config.experimental.get_visible_devices("GPU")[0]
tf.config.experimental.set_memory_growth(gpu, enable=True)


def vis_layer_with_depth(conv, deconv, i, img, queue):
    print("Enter function with depth:{}".format(i))
    img_, activation = vis_layer(conv, deconv, layer=i, image=img)
    queue.put([i, img_])
    return 1


if __name__ == "__main__":
    img_path = "./data/cat.jpg"
    img = load_images(img_path)
    conv = forward
    deconv = backward
    # mp.freeze_support()

    context = mp.get_context("spawn")
    manager = mp.Manager()
    q = manager.Queue()
    for i in range(1, 14):
        process_eval = context.Process(target=vis_layer_with_depth, args=(conv, deconv, i, img, q))
        process_eval.daemon = True
        process_eval.start()
        process_eval.join()
    print("13 layer reconstruction finished")
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))

    def plot_fig(i, img_):
        j = i - 1
        ax[j // 4, j % 4].imshow(img_)

    while not q.empty():
        ret = q.get()
        plot_fig(*ret)
    output_fig_path = './output/cat_vis_multi_layer.png'
    fig.savefig(output_fig_path)
    plt.show()

