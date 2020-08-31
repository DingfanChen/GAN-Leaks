import numpy as np
import os
import sys
import pickle
import argparse
from tqdm import tqdm
import torch

### import tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools/lpips_pytorch'))
from utils import *
import lpips_pytorch as ps
from LBFGS_pytorch import FullBatchLBFGS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../gan_models/vaegan'))
from train import *

### Hyperparameters
LAMBDA2 = 0.2
LAMBDA3 = 0.001
LBFGS_LR = 0.015
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
    parser.add_argument('--data_num', '-dnum', type=int, default=100,
                        help='the number of query images to be considered')
    parser.add_argument('--batch_size', '-bs', type=int, default=20,
                        help='batch size (should not be too large for better optimization performance)')
    parser.add_argument('--resolution', '-resolution', type=int, default=64,
                        help='generated image resolution')
    parser.add_argument('--initialize_type', '-init', type=str, default='random',
                        choices=['zero',  # 'zero': initialize the z to be zeros
                                 'random',  # 'random': use normal distributed initialization
                                 'nn',  # 'nn': use nearest-neighbor initialization
                                 ],
                        help='the initialization techniques')
    parser.add_argument('--nn_dir', '-ndir', type=str,
                        help='the directory for storing the fbb(KNN) results')
    parser.add_argument('--distance', '-dist', type=str, default='l2-lpips', choices=['l2', 'l2-lpips'],
                        help='the objective function type')
    parser.add_argument('--if_norm_reg', '-reg', action='store_true', default=True,
                        help='enable the norm regularizer')
    parser.add_argument('--maxfunc', '-mf', type=int, default=1000,
                        help='the maximum number of function calls (for scipy optimizer)')
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
    save_dir = os.path.join(os.path.dirname(__file__), 'results/wb', args.exp_name)
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
class LatentZ(torch.nn.Module):
    def __init__(self, init_val):
        super(LatentZ, self).__init__()
        self.z = torch.nn.Parameter(init_val.data)

    def forward(self):
        return self.z

    def reinit(self, init_val):
        self.z = torch.nn.Parameter(init_val.data)


class Loss(torch.nn.Module):
    def __init__(self, netG, distance, if_norm_reg=False, z_dim=100):
        super(Loss, self).__init__()
        self.distance = distance
        self.lpips_model = ps.PerceptualLoss()
        self.netG = netG
        self.if_norm_reg = if_norm_reg
        self.z_dim = z_dim

        ### loss
        if distance == 'l2':
            print('Use distance: l2')
            self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2, dim=[1, 2, 3])
            self.loss_lpips_fn = lambda x, y: 0.

        elif distance == 'l2-lpips':
            print('Use distance: lpips + l2')
            self.loss_lpips_fn = lambda x, y: self.lpips_model.forward(x, y, normalize=False).view(-1)
            self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2, dim=[1, 2, 3])

    def forward(self, z, x_gt):
        self.x_hat = self.netG(z)
        self.loss_lpips = self.loss_lpips_fn(self.x_hat, x_gt)
        self.loss_l2 = self.loss_l2_fn(self.x_hat, x_gt)
        self.vec_loss = LAMBDA2 * self.loss_lpips + self.loss_l2

        if self.if_norm_reg:
            z_ = z.view(-1, self.z_dim)
            norm = torch.sum(z_ ** 2, dim=1)
            norm_penalty = (norm - self.z_dim) ** 2
            self.vec_loss += LAMBDA3 * norm_penalty

        return self.vec_loss


def optimize_z_lbfgs(loss_model,
                     init_val,
                     query_imgs,
                     save_dir,
                     max_func):
    ### store results
    all_loss = []
    all_z = []
    all_x_hat = []

    ### run the optimization for all query data
    size = len(query_imgs)
    for i in tqdm(range(size // BATCH_SIZE)):
        save_dir_batch = os.path.join(save_dir, str(i))

        try:
            x_batch = query_imgs[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            x_gt = torch.from_numpy(x_batch).permute(0, 3, 1, 2).cuda()

            if os.path.exists(save_dir_batch):
                pass
            else:
                visualize_gt(x_batch, check_folder(save_dir_batch))

                ### initialize z
                z = Variable(torch.FloatTensor(init_val[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])).cuda()
                z_model = LatentZ(z)

                ### LBFGS optimizer
                optimizer = FullBatchLBFGS(z_model.parameters(), lr=LBFGS_LR, history_size=20, line_search='Wolfe',
                                           debug=False)

                ### optimize
                loss_progress = []

                def closure():
                    optimizer.zero_grad()
                    vec_loss = loss_model.forward(z_model.forward(), x_gt)
                    vec_loss_np = vec_loss.detach().cpu().numpy()
                    loss_progress.append(vec_loss_np)
                    final_loss = torch.mean(vec_loss)
                    return final_loss

                for step in range(max_func):
                    loss_model.forward(z_model.forward(), x_gt)
                    final_loss = closure()
                    final_loss.backward()

                    options = {'closure': closure, 'current_loss': final_loss, 'max_ls': 20}
                    obj, grad, lr, _, _, _, _, _ = optimizer.step(options)

                    if step == 0:
                        ### store init
                        x_hat_curr = loss_model.x_hat.data.cpu().numpy()
                        x_hat_curr = np.transpose(x_hat_curr, [0, 2, 3, 1])
                        vec_loss_curr = loss_model.vec_loss.data.cpu().numpy()
                        visualize_progress(x_hat_curr, vec_loss_curr, save_dir_batch, step)  # visualize init

                    if step == max_func - 1:
                        vec_loss_curr = loss_model.vec_loss.data.cpu().numpy()
                        z_curr = z_model.z.data.cpu().numpy()
                        x_hat_curr = loss_model.x_hat.data.cpu().numpy()
                        x_hat_curr = np.transpose(x_hat_curr, [0, 2, 3, 1])

                        loss_lpips = loss_model.loss_lpips.data.cpu().numpy()
                        loss_l2 = loss_model.loss_l2.data.cpu().numpy()
                        save_files(save_dir_batch, ['l2', 'lpips'], [loss_l2, loss_lpips])

                        ### store results
                        visualize_progress(x_hat_curr, vec_loss_curr, save_dir_batch, step)  # visualize finale
                        all_loss.append(vec_loss_curr)
                        all_z.append(z_curr)
                        all_x_hat.append(x_hat_curr)

                        save_files(save_dir_batch,
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

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    ### set up Generator
    network_path = os.path.join(load_dir, 'netG.pt')
    netG = torch.load(network_path).cuda()
    netG.eval()
    Z_DIM = netG.deconv1.module.in_channels
    resolution = args.resolution

    ### define loss
    loss_model = Loss(netG, args.distance, if_norm_reg=False, z_dim=Z_DIM)

    ### initialization
    if args.initialize_type == 'zero':
        init_val = np.zeros((args.data_num, Z_DIM, 1, 1))
        init_val_pos = init_val
        init_val_neg = init_val

    elif args.initialize_type == 'random':
        np.random.seed(RANDOM_SEED)
        init_val_np = np.random.normal(size=(Z_DIM, 1, 1))
        init_val_np = init_val_np / np.sqrt(np.mean(np.square(init_val_np)) + 1e-8)
        init_val = np.tile(init_val_np, (args.data_num, 1, 1, 1)).astype(np.float32)
        init_val_pos = init_val
        init_val_neg = init_val

    elif args.initialize_type == 'nn':
        idx = 0
        init_val_pos = np.load(os.path.join(args.nn_dir, 'pos_z.npy'))[:, idx, :]
        init_val_pos = np.reshape(init_val_pos, [len(init_val_pos), Z_DIM, 1, 1])
        init_val_neg = np.load(os.path.join(args.nn_dir, 'neg_z.npy'))[:, idx, :]
        init_val_neg = np.reshape(init_val_neg, [len(init_val_neg), Z_DIM, 1, 1])
    else:
        raise NotImplementedError

    ### positive ###
    pos_data_paths = get_filepaths_from_dir(args.pos_data_dir, ext='png')[: args.data_num]
    pos_query_imgs = np.array([read_image(f, resolution) for f in pos_data_paths])
    query_loss, query_z, query_xhat = optimize_z_lbfgs(loss_model,
                                                       init_val_pos,
                                                       pos_query_imgs,
                                                       check_folder(os.path.join(save_dir, 'pos_results')),
                                                       args.maxfunc)
    save_files(save_dir, ['pos_loss'], [query_loss])

    ### negative ###
    neg_data_paths = get_filepaths_from_dir(args.neg_data_dir, ext='png')[: args.data_num]
    neg_query_imgs = np.array([read_image(f, resolution) for f in neg_data_paths])
    query_loss, query_z, query_xhat = optimize_z_lbfgs(loss_model,
                                                       init_val_neg,
                                                       neg_query_imgs,
                                                       check_folder(os.path.join(save_dir, 'neg_results')),
                                                       args.maxfunc)
    save_files(save_dir, ['neg_loss'], [query_loss])


if __name__ == '__main__':
    main()
