import numpy as np
import os
import sys
import pickle
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
from utils import *
from sklearn.neighbors import NearestNeighbors

### Hyperparameters
K = 5
BATCH_SIZE = 10


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-name', type=str, required=True,
                        help='the name of the current experiment (used to set up the save_dir)')
    parser.add_argument('--gan_model_dir', '-gdir', type=str, required=True,
                        help='directory for the Victim GAN model (save the generated.npz file)')
    parser.add_argument('--pos_data_dir', '-posdir', type=str,
                        help='the directory for the positive (training) query images set')
    parser.add_argument('--neg_data_dir', '-negdir', type=str,
                        help='the directory for the negative (testing) query images set')
    parser.add_argument('--data_num', '-dnum', type=int, default=20000,
                        help='the number of query images to be considered')
    parser.add_argument('--resolution', '-resolution', type=int, default=64,
                        help='generated image resolution')
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
    save_dir = os.path.join(os.path.dirname(__file__), 'results/fbb', args.exp_name)
    check_folder(save_dir)

    ## store the parameters
    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.writelines(k + ":" + str(v) + "\n")
            print(k + ":" + str(v))
    pickle.dump(vars(args), open(os.path.join(save_dir, 'params.pkl'), 'wb'), protocol=2)
    return args, save_dir, args.gan_model_dir


#############################################################################################################
# main nearest neighbor search function
#############################################################################################################
def find_knn(nn_obj, query_imgs):
    '''
    :param nn_obj: Nearest Neighbor object
    :param query_imgs: query images
    :return:
        dist: distance between query samples to its KNNs among generated samples
        idx: index of the KNNs
    '''
    dist = []
    idx = []
    for i in tqdm(range(len(query_imgs) // BATCH_SIZE)):
        x_batch = query_imgs[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        x_batch = np.reshape(x_batch, [BATCH_SIZE, -1])
        dist_batch, idx_batch = nn_obj.kneighbors(x_batch, K)
        dist.append(dist_batch)
        idx.append(idx_batch)

    try:
        dist = np.concatenate(dist)
        idx = np.concatenate(idx)
    except:
        dist = np.array(dist)
        idx = np.array(idx)
    return dist, idx


def find_pred_z(gen_z, idx):
    '''
    :param gen_z: latent codes of the generated samples
    :param idx: index of the KNN
    :return:
        pred_z: predicted latent code
    '''
    pred_z = []
    for i in range(len(idx)):
        pred_z.append([gen_z[idx[i, nn]] for nn in range(K)])
    pred_z = np.array(pred_z)
    return pred_z


#############################################################################################################
# main
#############################################################################################################
def main():
    args, save_dir, load_dir = check_args(parse_arguments())
    resolution = args.resolution

    ### load generated samples
    generate = np.load(os.path.join(load_dir, 'generated.npz'))
    gen_imgs = generate['img_r01']
    gen_z = generate['noise']
    gen_feature = np.reshape(gen_imgs, [len(gen_imgs), -1])
    gen_feature = 2. * gen_feature - 1.

    ### load query images
    pos_data_paths = get_filepaths_from_dir(args.pos_data_dir, ext='png')[: args.data_num]
    pos_query_imgs = np.array([read_image(f, resolution) for f in pos_data_paths])

    neg_data_paths = get_filepaths_from_dir(args.neg_data_dir, ext='png')[: args.data_num]
    neg_query_imgs = np.array([read_image(f, resolution) for f in neg_data_paths])

    ### nearest neighbor search
    nn_obj = NearestNeighbors(K, n_jobs=16)
    nn_obj.fit(gen_feature)

    ### positive query
    pos_loss, pos_idx = find_knn(nn_obj, pos_query_imgs)
    pos_z = find_pred_z(gen_z, pos_idx)
    save_files(save_dir, ['pos_loss', 'pos_idx', 'pos_z'], [pos_loss, pos_idx, pos_z])

    ### negative query
    neg_loss, neg_idx = find_knn(nn_obj, neg_query_imgs)
    neg_z = find_pred_z(gen_z, neg_idx)
    save_files(save_dir, ['neg_loss', 'neg_idx', 'neg_z'], [neg_loss, neg_idx, neg_z])


if __name__ == '__main__':
    main()
