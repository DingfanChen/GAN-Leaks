# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import time
import numpy as np
import tensorflow as tf

import config
import misc
import tfutil


# ----------------------------------------------------------------------------
# Generate random images or image grids using a previously trained network.
def generate_fake_images(model_path, out_dir, num_samples, random_seed=1000, image_shrink=1, minibatch_size=32):
    random_state = np.random.RandomState(random_seed)

    network_pkl = model_path
    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(network_pkl)

    latents = misc.random_latents(num_samples, Gs, random_state)
    labels = np.zeros([latents.shape[0], 0], np.float32)
    images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5,
                    out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
    save_dir = misc.make_dir(out_dir)
    misc.save_image_grid(images[:100], os.path.join(save_dir, 'samples.png'), [0, 255], [10, 10])

    img_r01 = images.astype(np.float32) / 255.
    img_r01 = img_r01.transpose(0, 2, 3, 1)  # NCHW => NHWC
    np.savez_compressed(os.path.join(save_dir, 'generated.npz'), noise=latents, img_r01=img_r01)


# ----------------------------------------------------------------------------
# Evaluate one or more metrics for a previous training run.
# To run, uncomment one of the appropriate lines in config.py and launch train.py.

def evaluate_metrics(run_id, log, metrics, num_images, real_passes, minibatch_size=None):
    metric_class_names = {
        'swd': 'metrics.sliced_wasserstein.API',
        'fid': 'metrics.frechet_inception_distance.API',
        'is': 'metrics.inception_score.API',
        'msssim': 'metrics.ms_ssim.API',
    }

    # Locate training run and initialize logging.
    result_subdir = misc.locate_result_subdir(run_id)
    snapshot_pkls = misc.list_network_pkls(result_subdir, include_final=False)
    assert len(snapshot_pkls) >= 1
    log_file = os.path.join(result_subdir, log)
    print('Logging output to', log_file)
    misc.set_output_log_file(log_file)

    # Initialize dataset and select minibatch size.
    dataset_obj, mirror_augment = misc.load_dataset_for_previous_run(result_subdir, verbose=True, shuffle_mb=0)
    if minibatch_size is None:
        minibatch_size = np.clip(8192 // dataset_obj.shape[1], 4, 256)

    # Initialize metrics.
    metric_objs = []
    for name in metrics:
        class_name = metric_class_names.get(name, name)
        print('Initializing %s...' % class_name)
        class_def = tfutil.import_obj(class_name)
        image_shape = [3] + dataset_obj.shape[1:]
        obj = class_def(num_images=num_images, image_shape=image_shape, image_dtype=np.uint8,
                        minibatch_size=minibatch_size)
        tfutil.init_uninited_vars()
        mode = 'warmup'
        obj.begin(mode)
        for idx in range(10):
            obj.feed(mode, np.random.randint(0, 256, size=[minibatch_size] + image_shape, dtype=np.uint8))
        obj.end(mode)
        metric_objs.append(obj)

    # Print table header.
    print()
    print('%-10s%-12s' % ('Snapshot', 'Time_eval'), end='')
    for obj in metric_objs:
        for name, fmt in zip(obj.get_metric_names(), obj.get_metric_formatting()):
            print('%-*s' % (len(fmt % 0), name), end='')
    print()
    print('%-10s%-12s' % ('---', '---'), end='')
    for obj in metric_objs:
        for fmt in obj.get_metric_formatting():
            print('%-*s' % (len(fmt % 0), '---'), end='')
    print()

    # Feed in reals.
    for title, mode in [('Reals', 'reals'), ('Reals2', 'fakes')][:real_passes]:
        print('%-10s' % title, end='')
        time_begin = time.time()
        labels = np.zeros([num_images, dataset_obj.label_size], dtype=np.float32)
        [obj.begin(mode) for obj in metric_objs]
        for begin in range(0, num_images, minibatch_size):
            end = min(begin + minibatch_size, num_images)
            images, labels[begin:end] = dataset_obj.get_minibatch_np(end - begin)
            if mirror_augment:
                images = misc.apply_mirror_augment(images)
            if images.shape[1] == 1:
                images = np.tile(images, [1, 3, 1, 1])  # grayscale => RGB
            [obj.feed(mode, images) for obj in metric_objs]
        results = [obj.end(mode) for obj in metric_objs]
        print('%-12s' % misc.format_time(time.time() - time_begin), end='')
        for obj, vals in zip(metric_objs, results):
            for val, fmt in zip(vals, obj.get_metric_formatting()):
                print(fmt % val, end='')
        print()

    # Evaluate each network snapshot.
    for snapshot_idx, snapshot_pkl in enumerate(reversed(snapshot_pkls)):
        prefix = 'network-snapshot-';
        postfix = '.pkl'
        snapshot_name = os.path.basename(snapshot_pkl)
        assert snapshot_name.startswith(prefix) and snapshot_name.endswith(postfix)
        snapshot_kimg = int(snapshot_name[len(prefix): -len(postfix)])

        print('%-10d' % snapshot_kimg, end='')
        mode = 'fakes'
        [obj.begin(mode) for obj in metric_objs]
        time_begin = time.time()
        with tf.Graph().as_default(), tfutil.create_session(config.tf_config).as_default():
            G, D, Gs = misc.load_pkl(snapshot_pkl)
            for begin in range(0, num_images, minibatch_size):
                end = min(begin + minibatch_size, num_images)
                latents = misc.random_latents(end - begin, Gs)
                images = Gs.run(latents, labels[begin:end], num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5,
                                out_dtype=np.uint8)
                if images.shape[1] == 1:
                    images = np.tile(images, [1, 3, 1, 1])  # grayscale => RGB
                [obj.feed(mode, images) for obj in metric_objs]
        results = [obj.end(mode) for obj in metric_objs]
        print('%-12s' % misc.format_time(time.time() - time_begin), end='')
        for obj, vals in zip(metric_objs, results):
            for val, fmt in zip(vals, obj.get_metric_formatting()):
                print(fmt % val, end='')
        print()
    print()

# ----------------------------------------------------------------------------
