import os
import sys

import tensorflow as tf
from six.moves import urllib

_URL = 'http://rail.eecs.berkeley.edu/models/lpips'


def _download(url, output_dir):
    """Downloads the `url` file into `output_dir`.

    Modified from https://github.com/tensorflow/models/blob/master/research/slim/datasets/dataset_utils.py
    """
    filename = url.split('/')[-1]
    filepath = os.path.join(output_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')


def lpips(input0, input1, normalize=True, model='net-lin', net='alex', version=0.1):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) metric.

    Args:
        input0: An image tensor of shape `[..., height, width, channels]`,
            with values in [0, 1].
        input1: An image tensor of shape `[..., height, width, channels]`,
            with values in [0, 1].

    Returns:
        The Learned Perceptual Image Patch Similarity (LPIPS) distance.

    Reference:
        Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang.
        The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
        In CVPR, 2018.
    """
    # flatten the leading dimensions
    batch_shape = tf.shape(input0)[:-3]
    input0 = tf.reshape(input0, tf.concat([[-1], tf.shape(input0)[-3:]], axis=0))
    input1 = tf.reshape(input1, tf.concat([[-1], tf.shape(input1)[-3:]], axis=0))
    # NHWC to NCHW
    input0 = tf.transpose(input0, [0, 3, 1, 2])
    input1 = tf.transpose(input1, [0, 3, 1, 2])

    # normalize to [-1, 1]
    if normalize:
        input0 = input0 * 2.0 - 1.0
        input1 = input1 * 2.0 - 1.0

    input0_name, input1_name = '0:0', '1:0'

    default_graph = tf.get_default_graph()
    producer_version = default_graph.graph_def_versions.producer


    model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'pretrained_models')
    # files to try. try a specific producer version, but fallback to the version-less version (latest).
    pb_fnames = [
        '%s_%s_v%s_%d.pb' % (model, net, version, producer_version),
        '%s_%s_v%s.pb' % (model, net, version),
    ]
    for pb_fname in pb_fnames:
        if not os.path.isfile(os.path.join(model_dir, pb_fname)):
            try:
                _download(os.path.join(_URL, pb_fname), model_dir)
            except urllib.error.HTTPError:
                pass
        if os.path.isfile(os.path.join(model_dir, pb_fname)):
            break

    with open(os.path.join(model_dir, pb_fname), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def,
                                input_map={input0_name: input0, input1_name: input1})
        distance, = default_graph.get_operations()[-1].outputs

    if distance.shape.ndims == 4:
        distance = tf.squeeze(distance, axis=[-3, -2, -1])
    # reshape the leading dimensions
    distance = tf.reshape(distance, batch_shape)
    return distance

def get_lpips_scores(imgset0,imgset1,max_batch_size=64):

    import tqdm
    import numpy as np
    assert len(imgset0) == len(imgset1)
    image0_ph = tf.placeholder(tf.float32)
    image1_ph = tf.placeholder(tf.float32)
    dist = lpips(image0_ph, image1_ph, model='net-lin', net='vgg', version='0.1')
    sess = tf.Session()

    bs = max_batch_size
    all_score = []
    num_batch = len(imgset0) // bs
    for i in tqdm.tqdm(range(num_batch)):
        all_score.append(sess.run(dist, feed_dict={image0_ph: imgset0[i * bs:(i + 1) * bs],
                                                   image1_ph: imgset1[i * bs:(i + 1) * bs]}))
    all_score.append(
        sess.run(dist, feed_dict={image0_ph: imgset0[num_batch * bs:], image1_ph: imgset1[num_batch * bs:]}))
    try:
        all_score = np.concatenate(all_score)
    except:
        all_score = np.array(all_score)
    return all_score
