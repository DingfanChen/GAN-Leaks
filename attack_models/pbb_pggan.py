import os
import sys
import argparse
import pickle
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm

### import tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools/lpips_tensorflow'))
from utils import *
import lpips_tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../gan_models/pggan'))

### Hyperparameters
LAMBDA2 = 0.2
LAMBDA3 = 0.001
RANDOM_SEED = 1000


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-name', type=str, required=True,
                        help='the name of the current experiment (used to set up the save_dir)')
    parser.add_argument('--gan_model_dir', '-gdir', type=str, required=True,
                        help='directory for the Victim GAN model')
    parser.add_argument('--pos_data_dir', '-posdir', type=str,
                        help='the directory for the positive (training) query images set')
    parser.add_argument('--neg_data_dir', '-negdir', type=str,
                        help='the directory for the negative (testing) query images set')
    parser.add_argument('--data_num', '-dnum', type=int, default=5,
                        help='the number of query images to be considered')
    parser.add_argument('--batch_size', '-bs', type=int, default=1,
                        help='batch size')
    parser.add_argument('--initialize_type', '-init', type=str, default='zero',
                        choices=['zero',  # 'zero': initialize the z to be zeros
                                 'random',  # 'random': use normal distributed initialization
                                 'nn',  # 'nn' : use nearest-neighbor initialization
                                  ],
                        help='the initialization techniques')
    parser.add_argument('--nn_dir', '-ndir', type=str,
                        help='directory for the fbb(KNN) results')
    parser.add_argument('--distance', '-dist', type=str, default='l2-lpips', choices=['l2', 'l2-lpips'],
                        help='the objective function type')
    parser.add_argument('--if_norm_reg', '-reg', action='store_true', default=True,
                        help='enable the norm regularizer')
    parser.add_argument('--maxiter', type=int, default=10,
                        help='the maximum number of iterations')
    return parser.parse_args()


def check_args(args):
    '''
    check and store the arguments as well as set up the save_dir
    :param args: arguments
    :return:
    '''
    ## load dir
    assert os.path.exists(args.gan_model_dir)

    ## set up save_dir
    save_dir = os.path.join(os.path.dirname(__file__), 'results/pbb', args.exp_name)
    check_folder(save_dir)

    ## store the parameters
    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.writelines(k + ":" + str(v) + "\n")
            print(k + ":" + str(v))
    pickle.dump(vars(args), open(os.path.join(save_dir, 'params.pkl'), 'wb'), protocol=2)
    return args, save_dir, args.gan_model_dir


#############################################################################################################
# main optimization function
#############################################################################################################
def optimize_z(sess, z, x, x_hat,
               init_val_ph, init_val,
               query_imgs, save_dir,
               opt, vec_loss, vec_loss_dict):
    """
    z = argmin_z \lambda_1*|x_hat -x|^2  + \lambda_2 * LPIPS(x_hat,x)+ \lambda_3* L_reg
    where x_hat = G(z)

    :param sess:  session
    :param z:  latent variable
    :param x:  query
    :param x_hat: reconstruction
    :param init_val_ph: placeholder for initialization value
    :param init_val: dict that stores the initialization value
    :param query_imgs: query data
    :param save_dir:  save directory
    :param opt: optimization operator
    :param vec_loss: full loss
    :param vec_loss_dict: dict that stores each term in the objective
    :return:
    """

    ### store results
    all_loss = []
    all_z = []
    all_x_hat = []

    ### get the local variables
    vars = [var for var in tf.global_variables() if
            'latent_z' in var.name]
    for v in vars:
        print(v.name)

    ### callback function
    global step, loss_progress
    loss_progress = []
    step = 0

    def update(x_hat_curr, vec_loss_val):
        '''
        callback function for the lbfgs optimizer
        :param x_hat_curr:
        :param vec_loss_val:
        :return:
        '''
        global step, loss_progress
        loss_progress.append(vec_loss_val)
        step += 1

    ### run the optimization for all query data
    size = len(query_imgs)
    for i in tqdm(range(size // BATCH_SIZE)):
        save_dir_batch = os.path.join(save_dir, str(i))

        try:
            x_gt = query_imgs[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

            if os.path.exists(save_dir_batch):
                pass
            else:
                visualize_gt(x_gt, check_folder(save_dir_batch))

                ### initialize z
                if init_val_ph is not None:
                    sess.run(tf.initialize_variables(vars),
                             feed_dict={init_val_ph: init_val[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]})
                else:
                    sess.run(tf.initialize_variables(vars))

                ### optimize
                loss_progress = []
                step = 0
                batch_idx = i

                vec_loss_curr, z_curr, x_hat_curr = sess.run([vec_loss, z, x_hat], feed_dict={x: x_gt})
                visualize_progress(x_hat_curr, vec_loss_curr, save_dir_batch, step)  # visualize init
                opt.minimize(sess, feed_dict={x: x_gt}, fetches=[x_hat, vec_loss], loss_callback=update)
                vec_loss_curr, z_curr, x_hat_curr = sess.run([vec_loss, z, x_hat], feed_dict={x: x_gt})
                visualize_progress(x_hat_curr, vec_loss_curr, save_dir_batch, step)  # visualize final

                ### store results
                all_loss.append(vec_loss_curr)
                all_z.append(z_curr)
                all_x_hat.append(x_hat_curr)

                ### save to disk
                for key in vec_loss_dict.keys():
                    # each term in the objective
                    val = sess.run(vec_loss_dict[key], feed_dict={x: x_gt})
                    save_files(os.path.join(save_dir, str(i)), [key], [val])
                save_files(os.path.join(save_dir, str(i)),
                           ['full_loss', 'z', 'xhat', 'loss_progress'],
                           [vec_loss_curr, z_curr, x_hat_curr, np.array(loss_progress)])

        except KeyboardInterrupt:
            print('Stop optimization\n')
            break

    try:
        all_loss = np.concatenate(all_loss)
        all_z = np.concatenate(all_z)
        all_x_hat = np.concatenate(all_x_hat)
    except:
        all_loss = np.array(all_loss)
        all_z = np.array(all_z)
        all_x_hat = np.array(all_x_hat)
    return all_loss, all_z, all_x_hat


#############################################################################################################
# main
#############################################################################################################
def main():
    args, save_dir, load_dir = check_args(parse_arguments())

    ### open session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.as_default()

        ### load pre-trained model
        network_path = sorted(glob(os.path.join(load_dir, 'network-*.pkl')))[-1]
        with open(network_path, 'rb') as file:
            print('Loading networks from "%s"...' % network_path)
            G, D, Gs = pickle.load(file)
        Gs.print_layers()
        D.print_layers()

        ### define variables
        global BATCH_SIZE
        BATCH_SIZE = args.batch_size
        z_dim = G.input_shape[-1]
        resolution = G.output_shape[-1]
        x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, resolution, resolution, 3))
        labels = tf.zeros([BATCH_SIZE, 0], tf.float32)

        ### initialization
        init_val_ph = None
        init_val = {'pos': None, 'neg': None}
        if args.initialize_type == 'zero':
            z = tf.Variable(tf.zeros([BATCH_SIZE, z_dim], tf.float32), name='latent_z')

        elif args.initialize_type == 'random':
            np.random.seed(RANDOM_SEED)
            init_val_np = np.random.normal(size=(z_dim,))
            init_val_np = init_val_np / np.sqrt(np.mean(np.square(init_val_np)) + 1e-8)
            init = np.tile(init_val_np, (BATCH_SIZE, 1)).astype(np.float32)
            z = tf.Variable(init, name='latent_z')

        elif args.initialize_type == 'nn':
            init_val['pos'] = np.load(os.path.join(args.nn_dir, 'pos_z.npy'))[:, 0, :]
            init_val['neg'] = np.load(os.path.join(args.nn_dir, 'neg_z.npy'))[:, 0, :]
            init_val_ph = tf.placeholder(dtype=tf.float32, name='init_ph', shape=(BATCH_SIZE, z_dim))
            z = tf.Variable(init_val_ph, name='latent_z')

        else:
            raise NotImplementedError

        ### get the reconstruction (x_hat)
        with tf.variable_scope(Gs.scope, reuse=True):
            assert tf.get_variable_scope().name == Gs.scope
            with tf.control_dependencies(None):  # ignore surrounding control_dependencies
                inputs = [z, labels]
                x_hat = Gs._build_func(*inputs, is_template_graph=True, **Gs.static_kwargs)
                x_hat = tf.transpose(x_hat, perm=[0, 2, 3, 1])
                x_hat = tf.clip_by_value(x_hat, -1., 1.)

        ### loss
        if args.distance == 'l2':
            print('Use distance: l2')
            loss_l2 = tf.reduce_mean(tf.square(x_hat - x), axis=[1, 2, 3])
            vec_loss = loss_l2
            vec_losses = {'l2': loss_l2}

        elif args.distance == 'l2-lpips':
            print('Use distance: lpips + l2')
            loss_l2 = tf.reduce_mean(tf.square(x_hat - x), axis=[1, 2, 3])
            loss_lpips = lpips_tf.lpips(x_hat, x, normalize=False, model='net-lin', net='vgg', version='0.1')
            vec_losses = {'l2': loss_l2,
                          'lpips': loss_lpips}
            vec_loss = loss_l2 + LAMBDA2 * loss_lpips

        else:
            raise NotImplementedError

        ## regularizer
        norm = tf.reduce_sum(tf.square(z), axis=1)
        norm_penalty = (norm - z_dim) ** 2

        if args.if_norm_reg:
            loss = tf.reduce_mean(vec_loss) + LAMBDA3 * tf.reduce_mean(norm_penalty)
            vec_losses['norm'] = norm_penalty
        else:
            loss = tf.reduce_mean(vec_loss)

        ### set up optimizer
        opt = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                     var_list=[z],
                                                     method='Powell',
                                                     options={'maxiter': args.maxiter})

        ### load query images
        pos_data_paths = get_filepaths_from_dir(args.pos_data_dir, ext='png')[: args.data_num]
        pos_query_imgs = np.array([read_image(f, resolution) for f in pos_data_paths])

        neg_data_paths = get_filepaths_from_dir(args.neg_data_dir, ext='png')[: args.data_num]
        neg_query_imgs = np.array([read_image(f, resolution) for f in neg_data_paths])

        ### run the optimization on query images
        query_loss, query_z, query_xhat = optimize_z(sess, z, x, x_hat,
                                                     init_val_ph, init_val['pos'],
                                                     pos_query_imgs,
                                                     check_folder(os.path.join(save_dir, 'pos_results')),
                                                     opt, vec_loss, vec_losses)
        save_files(save_dir, ['pos_loss'], [query_loss])

        query_loss, query_z, query_xhat = optimize_z(sess, z, x, x_hat,
                                                     init_val_ph, init_val['neg'],
                                                     neg_query_imgs,
                                                     check_folder(os.path.join(save_dir, 'neg_results')),
                                                     opt, vec_loss, vec_losses)
        save_files(save_dir, ['neg_loss'], [query_loss])


if __name__ == '__main__':
    main()
