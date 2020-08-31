# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

# ----------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)

    def __getattr__(self, name): return self[name]

    def __setattr__(self, name, value): self[name] = value

    def __delattr__(self, name): del self[name]


# ----------------------------------------------------------------------------
# Paths.

data_dir = 'datasets'
result_dir = 'results/pggan_default'

# ----------------------------------------------------------------------------
# TensorFlow options.

tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
env = EasyDict()  # Environment variables, set by the main program in train.py.

tf_config[
    'graph_options.place_pruned_graph'] = True  # False (default) = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
tf_config[
    'gpu_options.allow_growth'] = True  # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
# env.CUDA_VISIBLE_DEVICES = '1'  # Unspecified (default) = Use all available GPUs. List of ints = CUDA device numbers to use.
env.TF_CPP_MIN_LOG_LEVEL = '1'  # 0 (default) = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.

# ----------------------------------------------------------------------------
# Official training configs, targeted mainly for CelebA-HQ.
# To run, comment/uncomment the lines as appropriate and launch train.py.

desc = 'pgan'  # Description string included in result subdir name.
random_seed = 1000  # Global random seed.
dataset = EasyDict()  # Options for dataset.load_dataset().
train = EasyDict(func='run.train_progressive_gan')  # Options for main training func.
G = EasyDict(func='networks.G_paper', latent_size=100, normalize_latents=False)  # Options for generator network.
D = EasyDict(func='networks.D_paper')  # Options for discriminator network.
G_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)  # Options for generator optimizer.
D_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)  # Options for discriminator optimizer.
G_loss = EasyDict(func='loss.G_wgan_acgan')  # Options for generator loss.
D_loss = EasyDict(func='loss.D_wgangp_acgan')  # Options for discriminator loss.
sched = EasyDict()  # Options for train.TrainingSchedule.
grid = EasyDict(size='640', layout='row_per_class')  # Options for train.setup_snapshot_image_grid().

# Dataset (choose one).
desc += '-celeba_train';
dataset = EasyDict(tfrecord_dir='celeba/train', resolution=64)

# # Config presets (choose one).
# desc += '-preset-v1-1gpu'; num_gpus = 1; D.mbstd_group_size = 16; sched.minibatch_base = 16; sched.minibatch_dict = {256: 14, 512: 6, 1024: 3}; sched.lod_training_kimg = 800; sched.lod_transition_kimg = 800; train.total_kimg = 19000
desc += '-preset-v2-1gpu'; num_gpus = 1; sched.minibatch_base = 4; sched.minibatch_dict = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4}; sched.G_lrate_dict = {1024: 0.0015}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); train.total_kimg = 12000
# desc += '-preset-v2-2gpus'; num_gpus = 2; sched.minibatch_base = 8; sched.minibatch_dict = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}; sched.G_lrate_dict = {512: 0.0015, 1024: 0.002}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); train.total_kimg = 12000
# # desc += '-preset-v2-4gpus'; num_gpus = 4; sched.minibatch_base = 16; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}; sched.G_lrate_dict = {256: 0.0015, 512: 0.002, 1024: 0.003}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); train.total_kimg = 12000
# # desc += '-preset-v2-8gpus'; num_gpus = 8; sched.minibatch_base = 32; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}; sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); train.total_kimg = 12000

# # Numerical precision (choose one).
# desc += '-fp32';
# sched.max_minibatch_per_gpu = {256: 8, 512: 8, 1024: 4}
# desc += '-fp16'; G.dtype = 'float16'; D.dtype = 'float16'; G.pixelnorm_epsilon=1e-4; G_opt.use_loss_scaling = True; D_opt.use_loss_scaling = True; sched.max_minibatch_per_gpu = {512: 16, 1024: 8}

# Disable individual features.
# desc += '-nogrowing'; sched.lod_initial_resolution = 1024; sched.lod_training_kimg = 0; sched.lod_transition_kimg = 0; train.total_kimg = 10000
# desc += '-nopixelnorm'; G.use_pixelnorm = False
# desc += '-nowscale'; G.use_wscale = False; D.use_wscale = False
# desc += '-noleakyrelu'; G.use_leakyrelu = False
# desc += '-nosmoothing'; train.G_smoothing = 0.0
# desc += '-norepeat'; train.minibatch_repeats = 1
# desc += '-noreset'; train.reset_opt_for_new_lod = False

# Special modes.
# desc += '-BENCHMARK'; sched.lod_initial_resolution = 4; sched.lod_training_kimg = 3; sched.lod_transition_kimg = 3; train.total_kimg = (8*2+1)*3; sched.tick_kimg_base = 1; sched.tick_kimg_dict = {}; train.image_snapshot_ticks = 1000; train.network_snapshot_ticks = 1000
# desc += '-BENCHMARK0'; sched.lod_initial_resolution = 1024; train.total_kimg = 10; sched.tick_kimg_base = 1; sched.tick_kimg_dict = {}; train.image_snapshot_ticks = 1000; train.network_snapshot_ticks = 1000
# desc += '-VERBOSE'; sched.tick_kimg_base = 1; sched.tick_kimg_dict = {}; train.image_snapshot_ticks = 1; train.network_snapshot_ticks = 100
# desc += '-GRAPH'; train.save_tf_graph = True
# desc += '-HIST'; train.save_weight_histograms = True

# ----------------------------------------------------------------------------
