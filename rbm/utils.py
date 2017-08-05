from scipy import misc
import tensorflow as tf
import numpy as np


def sample_prob(probs, rand):
    """ Takes a tensor of probabilities (as from a sigmoidal activation)
    and samples from all the distributions
    :param probs: tensor of probabilities
    :param rand: tensor (of the same shape as probs) of random values
    :return : binary sample of probabilities
    """
    return tf.nn.relu(tf.sign(probs - rand))

def gen_batches(data, batch_size):
    """ Divide input data into batches.
    :param data: input data
    :param batch_size: size of each batch
    :return: data divided into batches
    """
    data = np.array(data)

    for i in range(0, data.shape[0], batch_size):
        yield data[i:i+batch_size]

def gen_image(img, width, height, outfile, img_type='grey'):
    assert len(img) == width * height or len(img) == width * height * 3

    if img_type == 'grey':
        misc.imsave(outfile, img.reshape(width, height))

    elif img_type == 'color':
        misc.imsave(outfile, img.reshape(3, width, height))