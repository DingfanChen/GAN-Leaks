import os
import numpy as np
import tensorflow as tf
import argparse
import pickle

from train import *
from utils import *


####################################################################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, type=str,
                        help='Directory for saving the model checkpoints')
    parser.add_argument('--out_dir', type=str,
                        help='path for saving the generated data (default: save to model dir)')
    parser.add_argument('--num_samples', type=int, default=20000,
                        help='num of samples')
    return parser.parse_args()


####################################################################################################################

if __name__ == '__main__':
    ### load config
    args = parse_args()
    num_samples = args.num_samples
    model_dir = args.model_dir
    out_dir = args.out_dir
    save_dir = model_dir if out_dir is None else out_dir
    config_path = os.path.join(model_dir, 'params.pkl')
    BS = 100
    if os.path.exists(config_path):
        config = pickle.load(open(os.path.join(model_dir, 'params.pkl'), 'r'))
        OUTPUT_SIZE = config['OUTPUT_SIZE']
        GAN_TYPE = config['Architecture']
        Z_DIM = config['Z_DIM']
    else:
        OUTPUT_SIZE = 64
        GAN_TYPE = 'good'
        Z_DIM = 128

    ### set up the generator and the discriminator
    Generator, Discriminator = GeneratorAndDiscriminator(GAN_TYPE)

    ### define the varialbe for generating samples
    noise = tf.random_normal(shape=(BS, Z_DIM), dtype=tf.float32)
    samples = Generator(BS, is_training=False, noise=noise, z_dim=Z_DIM)

    ### set up session
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    ### set up saver for loading model
    load_saver = tf.train.Saver([v for v in tf.global_variables()])
    _, _ = load_model_from_checkpoint(model_dir, load_saver, sess)

    ### get samples
    noise_sample = []
    img_sample = []
    for i in range(int(np.ceil(num_samples / BS))):
        noise_batch, img_batch = sess.run([noise, samples])
        noise_sample.append(noise_batch)
        img_sample.append(img_batch)
    noise_sample = np.concatenate(noise_sample)[:num_samples]
    img_sample = np.concatenate(img_sample)[:num_samples]
    img_sample = np.reshape(img_sample, [-1, 3, OUTPUT_SIZE, OUTPUT_SIZE])
    save_image_grid(img_sample[:100], os.path.join(save_dir, 'samples.png'), [-1, 1], [10, 10])

    img_r01 = (img_sample + 1.) / 2.
    img_r01 = img_r01.transpose(0, 2, 3, 1)  # NCHW => NHWC
    np.savez_compressed(os.path.join(save_dir, 'generated.npz'),
                        noise=noise_sample,
                        img_r01=img_r01)
