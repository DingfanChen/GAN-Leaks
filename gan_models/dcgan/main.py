import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from model import DCGAN, gen_random
from utils import *

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train [25]")
flags.DEFINE_integer("iters", 50000, "maximum iterations to train")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("data_dir", "./data", "path to datasets [e.g. $HOME/data]")
flags.DEFINE_string("out_dir", "results", "Root directory for outputs [e.g. $HOME/out]")
flags.DEFINE_string("out_name", "dcgan_default",
                    "Folder (under out_root_dir) for all outputs. Generated automatically if left blank []")
flags.DEFINE_string("checkpoint_dir", "",
                    "Folder (under out_root_dir/out_name) to save checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Folder (under out_root_dir/out_name) to save samples [samples]")
flags.DEFINE_string("z_dist", "normal01", "'normal01' or 'uniform_unsigned' or uniform_signed")
flags.DEFINE_integer("max_to_keep", 1, "maximum number of checkpoints to keep")
flags.DEFINE_integer("sample_freq", 1000, "sample every this many iterations")
flags.DEFINE_integer("ckpt_freq", 200, "save checkpoint every this many iterations")
flags.DEFINE_integer("z_dim", 100, "dimensions of z")
flags.DEFINE_string("app", "train", " 'train' (training) or 'gen' (generation)")
flags.DEFINE_integer("num_samples", 20000, "number of generated samples")
FLAGS = flags.FLAGS


def main(_):
    ## expand user name and environment variables
    FLAGS.data_dir = expand_path(FLAGS.data_dir)
    FLAGS.out_dir = expand_path(FLAGS.out_dir)
    FLAGS.out_name = expand_path(FLAGS.out_name)
    FLAGS.checkpoint_dir = expand_path(FLAGS.checkpoint_dir)
    FLAGS.sample_dir = expand_path(FLAGS.sample_dir)
    ## output folders
    FLAGS.out_dir = os.path.join(FLAGS.out_dir, FLAGS.out_name)
    FLAGS.checkpoint_dir = FLAGS.out_dir if FLAGS.checkpoint_dir == "" else FLAGS.checkpoint_dir
    FLAGS.sample_dir = os.path.join(FLAGS.out_dir, FLAGS.sample_dir)

    if FLAGS.output_width is None: FLAGS.output_width = FLAGS.output_height

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(
            sess,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.batch_size,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            train_size=FLAGS.train_size,
            z_dim=FLAGS.z_dim,
            input_fname_pattern=FLAGS.input_fname_pattern,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir,
            data_dir=FLAGS.data_dir,
            out_dir=FLAGS.out_dir,
            max_to_keep=FLAGS.max_to_keep)

        show_all_variables()

        if FLAGS.app == 'train':
            if not os.path.exists(FLAGS.checkpoint_dir): os.makedirs(FLAGS.checkpoint_dir)
            if not os.path.exists(FLAGS.sample_dir): os.makedirs(FLAGS.sample_dir)

            ## save configs
            with open(os.path.join(FLAGS.out_dir, 'FLAGS.json'), 'w') as f:
                flags_dict = {k: FLAGS[k].value for k in FLAGS}
                json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
            pickle.dump(flags_dict, open(os.path.join(FLAGS.out_dir, 'params.pkl'), 'wb'), protocol=2)

            ## train
            dcgan.train(FLAGS)

        elif FLAGS.app == 'gen':
            ## load model
            assert os.path.exists(FLAGS.checkpoint_dir)
            load_success, load_counter = dcgan.load(FLAGS.checkpoint_dir)
            if not load_success:
                raise Exception("Checkpoint not found in " + FLAGS.checkpoint_dir)

            ## generate samples
            noise_sample = []
            img_sample = []
            for i in tqdm(range(int(np.ceil(FLAGS.num_samples / FLAGS.batch_size)))):
                z_batch = gen_random(FLAGS.z_dist, size=(FLAGS.batch_size, dcgan.z_dim))
                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_batch})
                noise_sample.append(z_batch)
                img_sample.append(samples)
            noise_sample = np.concatenate(noise_sample)[:FLAGS.num_samples]
            img_sample = np.concatenate(img_sample)[:FLAGS.num_samples]
            save_images(img_sample[:100], image_manifold_size(100),
                        os.path.join(FLAGS.checkpoint_dir, 'samples.png'))
            img_r01 = (img_sample + 1.) / 2.
            np.savez_compressed(os.path.join(FLAGS.checkpoint_dir, 'generated.npz'), noise=noise_sample,
                                img_r01=img_r01)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    tf.app.run()
