import os
import requests
import json
import tensorflow as tf

""" this file is utils lib """


def decode_predictions(preds, top=5):
    """
    Decode the prediction of an image net model.
    Arguments:
        preds: tensorflow tensor encoding a batch of predictions.
        top: integer, how many top-gueses of return 
    Return:
        A list of lists of top class predictions tuple,
        one list of turples per sample in batch in input
    """
    image_net_file = "./data/imagenet_class_index.json"
    web_file = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    class_index_dict = None
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError(
            "decode predictions expects a batch of prediction, (i.e a 2D array of shape (samples, 1000)) Found array with shape :"
            + str(preds.shape)
        )
    if os.path.exists(image_net_file):
        r = requests.get(web_file)
        with open(image_net_file, "w") as f:
            f.write(r.content)
    with open(image_net_file, "r") as f:
        class_index_dict = json.load(f)
    results = []
    for pred in preds:
        values, indices = tf.math.top_k(pred)
        result = [tuple(class_index_dict[str(i.numpy())]) + (pred[i].numpy(),) for i in indices]
        result = [tuple(class_index_dict[str(i.numpy())]) + (j.numpy(),) for i, j in zip(indices, values)]
        results.append(result)
    return results

