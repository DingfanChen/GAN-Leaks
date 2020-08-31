import os
import numpy as np
import argparse
import torch

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


##################################################################################################################
def main():
    args = parse_args()

    ### load model
    load_dir = args.model_dir
    network_path = os.path.join(load_dir, 'netG.pt')
    netG = torch.load(network_path)
    netG.eval()

    ### get config
    save_dir = load_dir if args.out_dir is None else args.out_dir
    num_samples = args.num_samples
    z_dim = netG.deconv1.module.in_channels
    batch_size = 100
    RANDOM_SEED = 1000
    torch.manual_seed(RANDOM_SEED)

    ### get samples
    noise_sample = []
    img_sample = []
    for i in range(int(np.ceil(num_samples / batch_size))):
        z = torch.randn(batch_size, z_dim, 1, 1).cuda()
        fake = netG(z)
        fake = fake.detach().cpu().data.numpy()
        img_sample.append(fake)
        noise_sample.append(z.detach().cpu().data.numpy())

    noise_sample = np.concatenate(noise_sample)[:num_samples]
    noise_sample = np.reshape(noise_sample, [-1, z_dim])
    img_sample = np.concatenate(img_sample)[:num_samples]
    save_image_grid(img_sample[:100], os.path.join(save_dir, 'samples.png'), [-1, 1], [10, 10])

    img_r01 = (img_sample + 1.) / 2.
    img_r01 = img_r01.transpose(0, 2, 3, 1)  # NCHW => NHWC
    np.savez_compressed(os.path.join(save_dir, 'generated.npz'),
                        noise=noise_sample,
                        img_r01=img_r01)


if __name__ == '__main__':
    main()
