import os
import sys
import functools
import numpy as np
import tensorflow as tf
import pickle
import argparse
import logging

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.ops.layernorm
from utils import *


####################################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--EXP_NAME', '-name', type=str, default='wgangp_default',
                        help='The name of the current experiment (used to set up the save_dir)')
    parser.add_argument('--data_dir', type=str,
                        help='Directory for the training data')
    parser.add_argument('--LOSS', type=str, choices=['dcgan', 'wgan-gp', 'wgan', 'lsgan'], default='wgan-gp',
                        help='Type of the GAN loss')
    parser.add_argument('--Architecture', '-arch', type=str, choices=['dcgan', 'resnet', 'fc', 'good'], default='good',
                        help='Type of the GAN architecture')
    parser.add_argument('--BATCH_SIZE', '-bs', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--ITERS', '-iters', type=int, default=100000,
                        help='Trainig iterations')
    parser.add_argument('--N_GPUS', '-gpu', type=int, default=1,
                        help='Number of GPUs')
    parser.add_argument('--DIM', '-dim', type=int, default=64,
                        help='Model dimensionality')
    parser.add_argument('--Z_DIM', '-z_dim', type=int, default=128,
                        help='Latent variable dimension')
    parser.add_argument('--OUTPUT_DIM', '-outdim', type=int, default=64 * 64 * 3,
                        help='Number of pixels in each image')
    parser.add_argument('--OUTPUT_SIZE', '-size', type=int, default=64,
                        help='The height and width of the output image')
    parser.add_argument('--CRITIC_ITERS', '-n_critic', type=int, default=5,
                        help='Critic steps per generator steps')
    parser.add_argument('--LAMBDA', '-lambda', type=float, default=10.,
                        help='Gradient penalty lambda hyperparameter')
    parser.add_argument('--MODE', '-mode', type=str, default='train',
                        choices=['train', 'pretrain'],
                        help='train a mode or compute the gradient norm(pretrain)')
    return check_args(parser.parse_args())


def check_args(args):
    ### set up save_dir
    save_dir = os.path.join('results', args.EXP_NAME)
    check_folder(save_dir)

    ### the argument dict
    arg_dict = vars(args)

    ### check argument
    if args.N_GPUS not in [1, 2]:
        raise Exception('Only 1 or 2 GPUs supported!')

    ### store the parameters
    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        for k, v in arg_dict.items():
            print(k + ":" + str(v) + "\n")
            f.writelines(k + ":" + str(v) + "\n")
    pickle.dump(arg_dict, open(os.path.join(save_dir, 'params.pkl'), 'wb'), protocol=2)
    return args, save_dir


####################################################################################################################
def GeneratorAndDiscriminator(architecture):
    """
    Choose which generator and discriminator architecture to use by
    uncommenting one of these lines.
    """
    if architecture == 'dcgan':
        # Baseline (G: DCGAN, D: DCGAN)
        return DCGANGenerator, DCGANDiscriminator

    elif architecture == 'resnet':
        # 101-layer ResNet G and D
        return ResnetGenerator, ResnetDiscriminator

    elif architecture == 'good':
        # For actually generating decent samples, use this one
        return GoodGenerator, GoodDiscriminator

    elif architecture == 'fc':
        # 512-dim 4-layer ReLU MLP G
        return FCGenerator, DCGANDiscriminator
    else:
        raise Exception('You must choose an architecture!')


#################################################################################################
def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = tflib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = tflib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)


def Normalize(name, axes, inputs, is_training, stats_iter=0.9, mode='wgan-gp'):
    if ('Discriminator' in name) and (mode == 'wgan-gp'):
        if axes != [0, 2, 3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return tflib.ops.layernorm.Layernorm(name, [1, 2, 3], inputs)
    else:
        return tflib.ops.batchnorm.Batchnorm(name, axes, inputs,
                                             is_training=is_training,
                                             stats_iter=stats_iter,
                                             fused=True)


def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)


def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4 * kwargs['output_dim']
    output = tflib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    return output


def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = tflib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return output


def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    output = tflib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    output = tflib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def BottleneckResidualBlock(name, input_dim, output_dim, filter_size, inputs, is_training=True, resample=None,
                            he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_shortcut = functools.partial(tflib.ops.conv2d.Conv2D, stride=2)
        conv_1 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim / 2)
        conv_1b = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim / 2, output_dim=output_dim / 2,
                                    stride=2)
        conv_2 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=output_dim / 2, output_dim=output_dim)
    elif resample == 'up':
        conv_shortcut = SubpixelConv2D
        conv_1 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim / 2)
        conv_1b = functools.partial(tflib.ops.deconv2d.Deconv2D, input_dim=input_dim / 2, output_dim=output_dim / 2)
        conv_2 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=output_dim / 2, output_dim=output_dim)
    elif resample == None:
        conv_shortcut = tflib.ops.conv2d.Conv2D
        conv_1 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim / 2)
        conv_1b = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim / 2, output_dim=output_dim / 2)
        conv_2 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim / 2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name + '.Conv1', filter_size=1, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_1b(name + '.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_2(name + '.Conv2', filter_size=1, inputs=output, he_init=he_init, biases=False)
    output = Normalize(name + '.BN', [0, 2, 3], output, is_training=is_training)

    return shortcut + (0.3 * output)


def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, is_training=True, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_shortcut = MeanPoolConv
        conv_1 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2 = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample == 'up':
        conv_shortcut = UpsampleConv
        conv_1 = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample == None:
        conv_shortcut = tflib.ops.conv2d.Conv2D
        conv_1 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name + '.BN1', [0, 2, 3], output, is_training=is_training)
    output = tf.nn.relu(output)
    output = conv_1(name + '.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    output = Normalize(name + '.BN2', [0, 2, 3], output, is_training=is_training)
    output = tf.nn.relu(output)
    output = conv_2(name + '.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

    return shortcut + output


#################################################################################################
# ! Generators
#################################################################################################
def GoodGenerator(n_samples, noise=None, z_dim=128, dim=64, output_dim=12288, is_training=True,
                  nonlinearity=tf.nn.relu):
    if noise is None:
        noise = tf.random_normal([n_samples, z_dim])

    output = tflib.ops.linear.Linear('Generator.Input', z_dim, 4 * 4 * 8 * dim, noise)
    output = tf.reshape(output, [-1, 8 * dim, 4, 4])

    output = ResidualBlock('Generator.Res1', 8 * dim, 8 * dim, 3, output, resample='up', is_training=is_training)
    output = ResidualBlock('Generator.Res2', 8 * dim, 4 * dim, 3, output, resample='up', is_training=is_training)
    output = ResidualBlock('Generator.Res3', 4 * dim, 2 * dim, 3, output, resample='up', is_training=is_training)
    output = ResidualBlock('Generator.Res4', 2 * dim, 1 * dim, 3, output, resample='up', is_training=is_training)

    output = Normalize('Generator.OutputN', [0, 2, 3], output, is_training=is_training)
    output = nonlinearity(output)
    output = tflib.ops.conv2d.Conv2D('Generator.Output', 1 * dim, 3, 3, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, output_dim])


def FCGenerator(n_samples, noise=None, z_dim=128, output_dim=12288, FC_DIM=512):
    if noise is None:
        noise = tf.random_normal([n_samples, z_dim])

    output = ReLULayer('Generator.1', z_dim, FC_DIM, noise)
    output = ReLULayer('Generator.2', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.3', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.4', FC_DIM, FC_DIM, output)
    output = tflib.ops.linear.Linear('Generator.Out', FC_DIM, output_dim, output)
    output = tf.tanh(output)
    return output


def DCGANGenerator(n_samples, noise=None, z_dim=128, dim=64, output_dim=12288, bn=True, is_training=True,
                   nonlinearity=tf.nn.relu):
    tflib.ops.conv2d.set_weights_stdev(0.02)
    tflib.ops.deconv2d.set_weights_stdev(0.02)
    tflib.ops.linear.set_weights_stdev(0.02)

    if noise is None:
        noise = tf.random_normal([n_samples, z_dim])

    output = tflib.ops.linear.Linear('Generator.Input', z_dim, 4 * 4 * 8 * dim, noise)
    output = tf.reshape(output, [-1, 8 * dim, 4, 4])
    if bn:
        output = Normalize('Generator.BN1', [0, 2, 3], output, is_training=is_training)
    output = nonlinearity(output)

    output = tflib.ops.deconv2d.Deconv2D('Generator.2', 8 * dim, 4 * dim, 5, output)
    if bn:
        output = Normalize('Generator.BN2', [0, 2, 3], output, is_training=is_training)
    output = nonlinearity(output)

    output = tflib.ops.deconv2d.Deconv2D('Generator.3', 4 * dim, 2 * dim, 5, output)
    if bn:
        output = Normalize('Generator.BN3', [0, 2, 3], output, is_training=is_training)
    output = nonlinearity(output)

    output = tflib.ops.deconv2d.Deconv2D('Generator.4', 2 * dim, dim, 5, output)
    if bn:
        output = Normalize('Generator.BN4', [0, 2, 3], output, is_training=is_training)
    output = nonlinearity(output)

    output = tflib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    tflib.ops.conv2d.unset_weights_stdev()
    tflib.ops.deconv2d.unset_weights_stdev()
    tflib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1, output_dim])


def WGANPaper_CrippledDCGANGenerator(n_samples, noise=None, z_dim=128, dim=64, output_dim=12288):
    if noise is None:
        noise = tf.random_normal([n_samples, z_dim])

    output = tflib.ops.linear.Linear('Generator.Input', z_dim, 4 * 4 * dim, noise)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, dim, 4, 4])

    output = tflib.ops.deconv2d.Deconv2D('Generator.2', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = tflib.ops.deconv2d.Deconv2D('Generator.3', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = tflib.ops.deconv2d.Deconv2D('Generator.4', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = tflib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, output_dim])


def ResnetGenerator(n_samples, noise=None, z_dim=128, dim=64, output_dim=12288):
    if noise is None:
        noise = tf.random_normal([n_samples, z_dim])

    output = tflib.ops.linear.Linear('Generator.Input', z_dim, 4 * 4 * 8 * dim, noise)
    output = tf.reshape(output, [-1, 8 * dim, 4, 4])

    for i in range(6):
        output = BottleneckResidualBlock('Generator.4x4_{}'.format(i), 8 * dim, 8 * dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up1', 8 * dim, 4 * dim, 3, output, resample='up')
    for i in range(6):
        output = BottleneckResidualBlock('Generator.8x8_{}'.format(i), 4 * dim, 4 * dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up2', 4 * dim, 2 * dim, 3, output, resample='up')
    for i in range(6):
        output = BottleneckResidualBlock('Generator.16x16_{}'.format(i), 2 * dim, 2 * dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up3', 2 * dim, 1 * dim, 3, output, resample='up')
    for i in range(6):
        output = BottleneckResidualBlock('Generator.32x32_{}'.format(i), 1 * dim, 1 * dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up4', 1 * dim, dim / 2, 3, output, resample='up')
    for i in range(5):
        output = BottleneckResidualBlock('Generator.64x64_{}'.format(i), dim / 2, dim / 2, 3, output, resample=None)

    output = tflib.ops.conv2d.Conv2D('Generator.Out', dim / 2, 3, 1, output, he_init=False)
    output = tf.tanh(output / 5.)

    return tf.reshape(output, [-1, output_dim])


def MultiplicativeDCGANGenerator(n_samples, noise=None, is_training=True, z_dim=128, dim=64, output_dim=12288, bn=True):
    if noise is None:
        noise = tf.random_normal([n_samples, z_dim])

    output = tflib.ops.linear.Linear('Generator.Input', z_dim, 4 * 4 * 8 * dim * 2, noise)
    output = tf.reshape(output, [-1, 8 * dim * 2, 4, 4])
    if bn:
        output = Normalize('Generator.BN1', [0, 2, 3], output, is_training=is_training)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = tflib.ops.deconv2d.Deconv2D('Generator.2', 8 * dim, 4 * dim * 2, 5, output)
    if bn:
        output = Normalize('Generator.BN2', [0, 2, 3], output, is_training=is_training)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = tflib.ops.deconv2d.Deconv2D('Generator.3', 4 * dim, 2 * dim * 2, 5, output)
    if bn:
        output = Normalize('Generator.BN3', [0, 2, 3], output, is_training=is_training)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = tflib.ops.deconv2d.Deconv2D('Generator.4', 2 * dim, dim * 2, 5, output)
    if bn:
        output = Normalize('Generator.BN4', [0, 2, 3], output, is_training=is_training)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = tflib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, output_dim])


########################################################################################################
# ! Discriminators
########################################################################################################
def GoodDiscriminator(inputs, output_size=64, dim=64, is_training=True):
    output = tf.reshape(inputs, [-1, 3, output_size, output_size])
    output = tflib.ops.conv2d.Conv2D('Discriminator.Input', 3, dim, 3, output, he_init=False)

    output = ResidualBlock('Discriminator.Res1', dim, 2 * dim, 3, output, resample='down', is_training=is_training)
    output = ResidualBlock('Discriminator.Res2', 2 * dim, 4 * dim, 3, output, resample='down', is_training=is_training)
    output = ResidualBlock('Discriminator.Res3', 4 * dim, 8 * dim, 3, output, resample='down', is_training=is_training)
    output = ResidualBlock('Discriminator.Res4', 8 * dim, 8 * dim, 3, output, resample='down', is_training=is_training)

    output = tf.reshape(output, [-1, 4 * 4 * 8 * dim])
    output = tflib.ops.linear.Linear('Discriminator.Output', 4 * 4 * 8 * dim, 1, output)

    return tf.reshape(output, [-1])


def MultiplicativeDCGANDiscriminator(inputs, output_size=64, dim=64, is_training=True, bn=True):
    output = tf.reshape(inputs, [-1, 3, output_size, output_size])

    output = tflib.ops.conv2d.Conv2D('Discriminator.1', 3, dim * 2, 5, output, stride=2)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = tflib.ops.conv2d.Conv2D('Discriminator.2', dim, 2 * dim * 2, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN2', [0, 2, 3], output, is_training=is_training)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = tflib.ops.conv2d.Conv2D('Discriminator.3', 2 * dim, 4 * dim * 2, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN3', [0, 2, 3], output, is_training=is_training)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = tflib.ops.conv2d.Conv2D('Discriminator.4', 4 * dim, 8 * dim * 2, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN4', [0, 2, 3], output, is_training=is_training)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = tf.reshape(output, [-1, 4 * 4 * 8 * dim])
    output = tflib.ops.linear.Linear('Discriminator.Output', 4 * 4 * 8 * dim, 1, output)

    return tf.reshape(output, [-1])


def ResnetDiscriminator(inputs, output_size=64, dim=64):
    output = tf.reshape(inputs, [-1, 3, output_size, output_size])
    output = tflib.ops.conv2d.Conv2D('Discriminator.In', 3, dim / 2, 1, output, he_init=False)

    for i in range(5):
        output = BottleneckResidualBlock('Discriminator.64x64_{}'.format(i), dim / 2, dim / 2, 3, output, resample=None)
    output = BottleneckResidualBlock('Discriminator.Down1', dim / 2, dim * 1, 3, output, resample='down')
    for i in range(6):
        output = BottleneckResidualBlock('Discriminator.32x32_{}'.format(i), dim * 1, dim * 1, 3, output, resample=None)
    output = BottleneckResidualBlock('Discriminator.Down2', dim * 1, dim * 2, 3, output, resample='down')
    for i in range(6):
        output = BottleneckResidualBlock('Discriminator.16x16_{}'.format(i), dim * 2, dim * 2, 3, output, resample=None)
    output = BottleneckResidualBlock('Discriminator.Down3', dim * 2, dim * 4, 3, output, resample='down')
    for i in range(6):
        output = BottleneckResidualBlock('Discriminator.8x8_{}'.format(i), dim * 4, dim * 4, 3, output, resample=None)
    output = BottleneckResidualBlock('Discriminator.Down4', dim * 4, dim * 8, 3, output, resample='down')
    for i in range(6):
        output = BottleneckResidualBlock('Discriminator.4x4_{}'.format(i), dim * 8, dim * 8, 3, output, resample=None)

    output = tf.reshape(output, [-1, 4 * 4 * 8 * dim])
    output = tflib.ops.linear.Linear('Discriminator.Output', 4 * 4 * 8 * dim, 1, output)

    return tf.reshape(output / 5., [-1])


def FCDiscriminator(inputs, output_dim=12288, FC_DIM=512, n_layers=3):
    output = LeakyReLULayer('Discriminator.Input', output_dim, FC_DIM, inputs)
    for i in range(n_layers):
        output = LeakyReLULayer('Discriminator.{}'.format(i), FC_DIM, FC_DIM, output)
    output = tflib.ops.linear.Linear('Discriminator.Out', FC_DIM, 1, output)

    return tf.reshape(output, [-1])


def DCGANDiscriminator(inputs, output_size=64, dim=64, bn=True, is_training=True, nonlinearity=LeakyReLU):
    output = tf.reshape(inputs, [-1, 3, output_size, output_size])

    tflib.ops.conv2d.set_weights_stdev(0.02)
    tflib.ops.deconv2d.set_weights_stdev(0.02)
    tflib.ops.linear.set_weights_stdev(0.02)

    output = tflib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
    output = nonlinearity(output)

    output = tflib.ops.conv2d.Conv2D('Discriminator.2', dim, 2 * dim, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN2', [0, 2, 3], output, is_training=is_training)
    output = nonlinearity(output)

    output = tflib.ops.conv2d.Conv2D('Discriminator.3', 2 * dim, 4 * dim, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN3', [0, 2, 3], output, is_training=is_training)
    output = nonlinearity(output)

    output = tflib.ops.conv2d.Conv2D('Discriminator.4', 4 * dim, 8 * dim, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN4', [0, 2, 3], output, is_training=is_training)
    output = nonlinearity(output)

    output = tf.reshape(output, [-1, 4 * 4 * 8 * dim])
    output = tflib.ops.linear.Linear('Discriminator.Output', 4 * 4 * 8 * dim, 1, output)

    tflib.ops.conv2d.unset_weights_stdev()
    tflib.ops.deconv2d.unset_weights_stdev()
    tflib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1])


############################################################################################################################
def main():
    ### get arguments
    args, SAVE_DIR = parse_arguments()
    data_dir = args.data_dir
    LOSS = args.LOSS
    BATCH_SIZE = args.BATCH_SIZE
    N_GPUS = args.N_GPUS
    OUTPUT_DIM = args.OUTPUT_DIM
    LAMBDA = args.LAMBDA
    SIZE = args.OUTPUT_SIZE
    Z_DIM = args.Z_DIM
    CRITIC_ITERS = args.CRITIC_ITERS
    ITERS = args.ITERS
    DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

    ### set up log file
    logging.basicConfig(filename=os.path.join(SAVE_DIR, 'out.log'), level=logging.INFO)
    logger = logging.getLogger(__name__)

    ### set up the generator and the discriminator
    Generator, Discriminator = GeneratorAndDiscriminator(args.Architecture)

    ### create session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

        ### load data
        def get_tfdataset_train(data_paths):
            dataset = tf.data.Dataset.from_tensor_slices(data_paths)
            get_image_tf = lambda x: tf.py_func(lambda f: read_image(f,
                                                                     resolution=SIZE).astype(np.float32), [x],
                                                [tf.float32])

            try:
                dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=100 * BATCH_SIZE))
                dataset = dataset.prefetch(buffer_size=10 * BATCH_SIZE)
                dataset = dataset.map(map_func=get_image_tf, num_parallel_calls=16)
                dataset = dataset.batch(BATCH_SIZE)

            except:
                dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=100 * BATCH_SIZE))
                dataset = dataset.prefetch(buffer_size=10 * BATCH_SIZE)
                dataset = dataset.apply(tf.data.experimental.map_and_batch(
                    map_func=get_image_tf, batch_size=BATCH_SIZE, num_parallel_calls=16))

            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            next_batch = tf.squeeze(iterator.get_next())
            next_batch.set_shape((BATCH_SIZE, SIZE, SIZE, 3))
            return dataset, iterator, next_batch

        data_paths = get_filepaths_from_dir(data_dir, ext='png')
        print('Load Data from %s' % data_dir)
        print('Population size: %d' % len(data_paths))
        logger.info('Load Data from %s' % data_dir)
        logger.info('Population size: %d' % len(data_paths))

        if args.MODE == 'train':
            dataset, iterator, next_batch = get_tfdataset_train(data_paths)
            session.run(iterator.make_initializer(dataset))
            all_real_data_conv = tf.transpose(next_batch, perm=[0, 3, 1, 2])
        else:
            raise NotImplementedError

        if tf.__version__.startswith('1.'):
            split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
        else:
            split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)

        gen_costs, disc_costs = [], []
        for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
            with tf.device(device):

                real_data = tf.reshape(real_data_conv, [BATCH_SIZE // len(DEVICES), OUTPUT_DIM])
                fake_data = Generator(BATCH_SIZE // len(DEVICES), z_dim=Z_DIM)

                disc_real = Discriminator(real_data, output_size=SIZE)
                disc_fake = Discriminator(fake_data, output_size=SIZE)

                if LOSS == 'wgan':
                    gen_cost = -tf.reduce_mean(disc_fake)
                    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                elif LOSS == 'wgan-gp':
                    gen_cost = -tf.reduce_mean(disc_fake)
                    disc_cost_ = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                    alpha = tf.random_uniform(
                        shape=[BATCH_SIZE // len(DEVICES), 1],
                        minval=0.,
                        maxval=1.
                    )
                    differences = fake_data - real_data
                    interpolates = real_data + (alpha * differences)
                    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                    gradient_penalty_vec = (slopes - 1.) ** 2
                    gradient_penalty = tf.reduce_mean(gradient_penalty_vec)
                    disc_cost = disc_cost_ + LAMBDA * gradient_penalty

                elif LOSS == 'dcgan':
                    try:  # tf pre-1.0 (bottom) vs 1.0 (top)
                        gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                          labels=tf.ones_like(
                                                                                              disc_fake)))
                        disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                           labels=tf.zeros_like(
                                                                                               disc_fake)))
                        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                                            labels=tf.ones_like(
                                                                                                disc_real)))
                    except Exception as e:
                        gen_cost = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
                        disc_cost = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
                        disc_cost += tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
                    disc_cost /= 2.

                elif LOSS == 'lsgan':
                    gen_cost = tf.reduce_mean((disc_fake - 1) ** 2)
                    disc_cost = (tf.reduce_mean((disc_real - 1) ** 2) + tf.reduce_mean((disc_fake - 0) ** 2)) / 2.

                else:
                    raise Exception()

                gen_costs.append(gen_cost)
                disc_costs.append(disc_cost)

        gen_cost = tf.add_n(gen_costs) / len(DEVICES)
        disc_cost = tf.add_n(disc_costs) / len(DEVICES)

        if LOSS == 'wgan':
            gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost,
                                                                                  var_list=lib.params_with_name(
                                                                                      'Generator'),
                                                                                  colocate_gradients_with_ops=True)
            disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost,
                                                                                   var_list=lib.params_with_name(
                                                                                       'Discriminator.'),
                                                                                   colocate_gradients_with_ops=True)

            clip_ops = []
            for var in lib.params_with_name('Discriminator'):
                clip_bounds = [-.01, .01]
                clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
            clip_disc_weights = tf.group(*clip_ops)

        elif LOSS == 'wgan-gp':
            gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(gen_cost,
                                                                                                    var_list=lib.params_with_name(
                                                                                                        'Generator'),
                                                                                                    colocate_gradients_with_ops=True)
            disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(disc_cost,
                                                                                                     var_list=lib.params_with_name(
                                                                                                         'Discriminator.'),
                                                                                                     colocate_gradients_with_ops=True)

        elif LOSS == 'dcgan':
            gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
                                                                                          var_list=lib.params_with_name(
                                                                                              'Generator'),
                                                                                          colocate_gradients_with_ops=True)
            disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
                                                                                           var_list=lib.params_with_name(
                                                                                               'Discriminator.'),
                                                                                           colocate_gradients_with_ops=True)

        elif LOSS == 'lsgan':
            gen_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(gen_cost,
                                                                                  var_list=lib.params_with_name(
                                                                                      'Generator'),
                                                                                  colocate_gradients_with_ops=True)
            disc_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(disc_cost,
                                                                                   var_list=lib.params_with_name(
                                                                                       'Discriminator.'),
                                                                                   colocate_gradients_with_ops=True)
        else:
            raise Exception()

        ## For generating samples
        fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, Z_DIM)).astype('float32'))
        all_fixed_noise_samples_test = []

        for device_index, device in enumerate(DEVICES):
            n_samples = int(BATCH_SIZE / len(DEVICES))
            all_fixed_noise_samples_test.append(
                Generator(n_samples, noise=fixed_noise[device_index * n_samples:(device_index + 1) * n_samples],
                          is_training=False, z_dim=Z_DIM))

        if tf.__version__.startswith('1.'):
            all_fixed_noise_samples_test = tf.concat(all_fixed_noise_samples_test, axis=0)
        else:
            all_fixed_noise_samples_test = tf.concat(0, all_fixed_noise_samples_test)

        def generate_image(iteration):
            samples_test = session.run(all_fixed_noise_samples_test)
            save_image_grid(samples_test.reshape((BATCH_SIZE, 3, SIZE, SIZE)),
                            os.path.join(SAVE_DIR, 'samples_test_{}.png'.format(iteration)),
                            drange=[-1, 1])

        ## Save a batch of ground-truth samples
        _x_r = session.run(next_batch)
        _x_r = np.transpose(_x_r, (0, 3, 1, 2))
        save_image_grid(_x_r.reshape((BATCH_SIZE, 3, SIZE, SIZE)),
                        os.path.join(SAVE_DIR, 'samples_groundtruth.png'),
                        drange=[-1, 1])

        ### Train loop
        try:
            session.run(tf.initialize_all_variables())
        except:
            session.run(tf.global_variables_initializer())

        ### set up saver
        saver = tf.train.Saver(max_to_keep=1)
        _, counter = load_model_from_checkpoint(SAVE_DIR, saver, session)

        for iteration in range(counter, ITERS):
            try:
                # Train generator
                if iteration > 0:
                    session.run(gen_train_op)

                # Train critic
                if (LOSS == 'dcgan') or (LOSS == 'lsgan'):
                    disc_iters = 1
                else:
                    disc_iters = CRITIC_ITERS

                for i in range(disc_iters):
                    _disc_cost, _ = session.run([disc_cost, disc_train_op])
                    if LOSS == 'wgan':
                        _ = session.run([clip_disc_weights])

                if iteration % 500 == 0:
                    generate_image(iteration)

                if (iteration < 5) or (iteration % 200 == 0):
                    print("iter {}, disc_cost {}".format(iteration, _disc_cost))

                if iteration % 500 == 0:
                    print('Saving model...')
                    saver.save(session, os.path.join(SAVE_DIR, 'checkpoint-') + str(iteration))
                    saver.export_meta_graph(os.path.join(SAVE_DIR, 'checkpoint-') + str(iteration) + '.meta')

            except KeyboardInterrupt:
                model_filename = os.path.join(SAVE_DIR, 'checkpoint-') + str(iteration)
                print('Stop training, saving model to %s' % model_filename)
                saver.save(session, model_filename)
                saver.export_meta_graph(model_filename + '.meta')
                break


if __name__ == '__main__':
    main()
