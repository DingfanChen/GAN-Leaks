import os
import sys
import random
import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from ops import *
from utils import *


######################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-name', type=str, default='vaegan_default',
                        help='the name of the current experiment (used to set up the save_dir)')
    parser.add_argument('--data_dir', '-dir', type=str,
                        help='directory for the training data')
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--nepoch', type=int, default=1000)
    return check_args(parser.parse_args())


def check_args(args):
    ### set up save_dir
    save_dir = os.path.join('results', args.exp_name)
    check_folder(save_dir)

    ### the argument dict
    arg_dict = vars(args)

    ### store the parameters
    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        for k, v in arg_dict.items():
            print(k + ":" + str(v) + "\n")
            f.writelines(k + ":" + str(v) + "\n")
    pickle.dump(arg_dict, open(os.path.join(save_dir, 'params.pkl'), 'wb'), protocol=2)
    return args, save_dir


######################################################################################################
## models
######################################################################################################
class Encoder(nn.Module):
    def __init__(self, z_dim, d=64):
        super(Encoder, self).__init__()

        self.cv1 = nn.Conv2d(3, d, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(d)

        self.cv2 = nn.Conv2d(d, d * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(d * 2)

        self.cv3 = nn.Conv2d(d * 2, d * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(d * 4)

        self.cv4 = nn.Conv2d(d * 4, d * 8, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(d * 8)

        self.relu = nn.ReLU()

        self.fc1 = ChannelsToLinear(512 * 4 * 4, 4 * z_dim)
        self.fc1_1 = nn.Linear(4 * z_dim, z_dim)
        self.bn6 = nn.BatchNorm1d(4 * z_dim)

        self.fc2 = ChannelsToLinear(512 * 4 * 4, 4 * z_dim)
        self.fc2_1 = nn.Linear(4 * z_dim, z_dim)
        self.bn7 = nn.BatchNorm1d(4 * z_dim)

    def encode(self, x):  # Q(z|x)

        h1 = self.relu(self.bn1(self.cv1(x)))
        h2 = self.relu(self.bn2(self.cv2(h1)))
        h3 = self.relu(self.bn3(self.cv3(h2)))
        h4 = self.relu(self.bn4(self.cv4(h3)))

        z_mu = self.fc1_1(self.relu(self.bn6(self.fc1(h4))))
        z_var = self.fc2_1(self.relu(self.bn7(self.fc2(h4))))
        return z_mu, z_var

    def reparametrize(self, mu, logvar):
        std = logvar.mul(1.0).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std) + mu

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparametrize(mu, logvar)
        return z


class Generator(nn.Module):
    def __init__(self, z_dim, d=64):
        super(Generator, self).__init__()
        self.deconv1 = SpectralNorm(nn.ConvTranspose2d(z_dim, d * 8, 4, 1, 0))
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = SpectralNorm(
            nn.ConvTranspose2d(d * 8, d * 4, kernel_size=4, stride=2, padding=1, output_padding=0))
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = SpectralNorm(
            nn.ConvTranspose2d(d * 4, d * 2, kernel_size=4, stride=2, padding=1, output_padding=0))
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = SpectralNorm(nn.ConvTranspose2d(d * 2, d, kernel_size=4, stride=2, padding=1, output_padding=0))
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.sa1 = SelfAttention(d * 2, 'relu')

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x, _ = self.sa1(F.relu(self.deconv3_bn(self.deconv3(x))))
        x = (F.relu(self.deconv4_bn(self.deconv4(x))))
        x = F.tanh(self.deconv5(x))
        return x


class DiscriminatorL(nn.Module):
    def __init__(self, z_dim=100, ngpu=1):
        super(DiscriminatorL, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            (nn.Linear(z_dim, 750)),
            nn.LeakyReLU(0.01, inplace=True),

            (nn.Linear(750, 750)),
            nn.LeakyReLU(0.01, inplace=True),

            (nn.Linear(750, 1)),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class Discriminator(nn.Module):
    def __init__(self, d=64):
        super(Discriminator, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(3, d, kernel_size=4, stride=2, padding=1))  # 32
        self.conv2 = SpectralNorm(nn.Conv2d(d, d * 2, kernel_size=4, stride=2, padding=1))  # 16
        self.conv3 = SpectralNorm(nn.Conv2d(d * 2, d * 4, kernel_size=4, stride=2, padding=1))  # 8
        self.conv4 = SpectralNorm(nn.Conv2d(d * 4, d * 8, kernel_size=4, stride=2, padding=1))  # 4
        self.conv5 = SpectralNorm(nn.Conv2d(d * 8, 1, kernel_size=4, stride=2, padding=1))  # 2
        self.sa1 = SelfAttention(d * 4, 'relu')
        self.fc = SpectralNorm(nn.Linear(4, 1))

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.02)
        x = F.leaky_relu((self.conv2(x)), 0.02)
        x, _ = self.sa1(F.leaky_relu((self.conv3(x)), 0.02))
        x = (F.leaky_relu((self.conv4(x)), 0.02))
        x = self.conv5(x)
        x = x.view(-1, 4)
        x = self.fc(x)
        return x.view(-1, 1).squeeze(1)


#############################################################################################################
def criterionG(output, label, real, fake, outputN, batchSize):
    '''
    G loss
    :param output: output of Discriminator
    :param label: real/fake label
    :param real:  real image
    :param fake:  fake image
    :param outputN: output of DiscriminatorL
    :param batchSize: batch size
    :return:
    '''

    ### reconstruction loss
    real_ = real.view(batchSize, 1, -1)
    fake = fake.view(batchSize, 1, -1)
    min_vals, min_idxs = torch.min(torch.abs(real_ - fake).sum(dim=2), dim=1)
    rec_pd = min_vals.sum()
    REC = (rec_pd) / batchSize

    ### wasserstein distance loss
    output = output.view(batchSize, 1)
    output = torch.max(output, dim=1)[0]
    BCE1 = (output).mean()

    ### cross entropy loss (DiscriminatorL: on the latent variable)
    bceloss = nn.BCELoss()
    BCE2 = bceloss(outputN, label)
    return 1.0 * BCE1 + 1.0 * BCE2 + 0.005 * REC


#############################################################################################################
class CelebaDataseat(Dataset):
    def __init__(self, data_dir, resolution=64, ext='png'):
        self.data_dir = data_dir
        self.resolution = resolution
        self.data_paths = get_filepaths_from_dir(self.data_dir, ext=ext)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        filepath = self.data_paths[index]
        return self.to_tensor(read_image(filepath, resolution=self.resolution))


#############################################################################################################
def main():
    args, save_dir = parse_arguments()

    ngpu = args.ngpu
    z_dim = args.z_dim
    batchSize = args.batch_size
    imageSize = args.image_size
    nepoch = args.nepoch
    data_dir = args.data_dir
    outf_img = check_folder(os.path.join(save_dir, 'img'))
    beta1 = 0.0
    cuda = True
    cudnn.benchmark = True
    device = torch.device("cuda:0" if cuda else "cpu")

    ## set seed
    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    ### load data
    dataset = CelebaDataseat(data_dir=data_dir, resolution=imageSize)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True,
                                             num_workers=int(10), drop_last=True)
    dataloader_iterator = iter(dataloader)
    print('Size of the training set: ', len(dataset))

    ### set up models
    netE = Encoder(z_dim=z_dim).to(device)
    netE.apply(weights_init)
    netG = Generator(z_dim=z_dim).to(device)
    netDl = DiscriminatorL(z_dim=z_dim, ngpu=ngpu).to(device)  # discriminator(on the latent variable)
    netD = Discriminator().to(device)  # discriminator(on the image)

    ### define the losses
    criterion = nn.BCELoss()
    real_label = 1
    fake_label = 0
    fixed_noise = torch.randn(batchSize, z_dim, 1, 1, device=device)

    ### setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=0.0004, betas=(beta1, 0.9))
    optimizerDl = optim.Adam(netDl.parameters(), lr=0.0002, betas=(beta1, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(beta1, 0.9))
    optimizerE = optim.Adam(netE.parameters(), lr=0.0001, betas=(beta1, 0.9))

    ### load previous trained model
    if_load, counter, checkpoint = load(save_dir)
    if if_load:
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netE.load_state_dict(checkpoint['netE_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        netDl.load_state_dict(checkpoint['netDl_state_dict'])
        optimizerG.load_state_dict(checkpoint['optG_state_dict'])
        optimizerE.load_state_dict(checkpoint['optE_state_dict'])
        optimizerD.load_state_dict(checkpoint['optD_state_dict'])
        optimizerDl.load_state_dict(checkpoint['optDl_state_dict'])

    for epoch in range(counter, nepoch):
        for i in range(5000 // (batchSize)):
            ############################
            # (1) Update Dl network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            for _ in range(1):
                netDl.zero_grad()

                # train with real
                try:
                    data = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(dataloader)
                    data = next(dataloader_iterator)
                real_ = data.to(device)
                label = torch.full((batchSize,), fake_label, device=device)

                output = netE(real_)
                output = netDl(output)
                errDl_real = criterion(output, label)
                errDl_real.backward()
                errDl = errDl_real

                # train with fake
                noise = torch.randn(batchSize, z_dim, device=device)
                label = torch.full((batchSize,), real_label, device=device)
                output = netDl(noise)
                errDl_real = criterion(output, label)
                errDl_real.backward()
                errDl += errDl_real

                optimizerDl.step()

            ############################
            # (2) Update D network: Hinge loss
            ###########################
            for _ in range(2):
                netD.zero_grad()

                # train with real
                try:
                    data = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(dataloader)
                    data = next(dataloader_iterator)
                real_ = data.to(device)
                out_real = netD(real_)

                noise = torch.randn(batchSize, z_dim, 1, 1, device=device)
                fake = netG(noise)
                out_fake = netD(fake.detach())

                errD_real = (nn.ReLU()(0.5 + out_real)).mean()
                errD_real.backward()
                errD_fake = (nn.ReLU()(0.5 - out_fake)).mean()
                errD_fake.backward()
                errD = errD_real + errD_fake

                optimizerD.step()

            ############################
            # (3) Update G & E network: maximize log(D(G(z)))
            ###########################
            for _ in range(1):
                netG.zero_grad()
                netE.zero_grad()

                try:
                    data = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(dataloader)
                    data = next(dataloader_iterator)

                real_ = data.to(device)
                real_ = real_.unsqueeze(1).repeat(1, 1, 1, 1, 1)
                real_ = real_.view(batchSize * 1, 3, imageSize, imageSize)

                encoded = netE(real_)
                fake_noise = encoded
                encoded = encoded.view(batchSize * 1, z_dim, 1, 1)

                rec_fake = netG(encoded)
                output = netD(rec_fake)
                outputN = netDl(fake_noise)

                label = torch.full((batchSize * 1,), real_label, device=device)
                errG = criterionG(output, label, real_, rec_fake, outputN, batchSize)
                errG.backward()
                optimizerG.step()
                optimizerE.step()

            if i % 100 == 0:
                print(
                    '[%d/%d][%d] Loss_D: %.4f, Loss_Dfake: %.4f, Loss_Dreal: %.4f, Loss_Dl: %.4f, Loss_G: %.4f'
                    % (epoch, nepoch, i, errD.item(), errD_fake.item(), errD_real.item(), errDl.item(), errG.item()))

        if epoch % 20 == 0:
            noise = fixed_noise
            fake = netG(noise)
            fake = fake.view(batchSize, 3, imageSize, imageSize)

            vutils.save_image(fake.detach(), '%s/epoch_%04d.png' % (outf_img, epoch), normalize=True)
            vutils.save_image(real_.detach(), '%s/real_%04d.png' % (outf_img, epoch), normalize=True)
            vutils.save_image(rec_fake.detach(), '%s/reconst_%04d.png' % (outf_img, epoch), normalize=True)

        if epoch % 10 == 0:
            save_dict = {'steps': epoch,
                         'netE_state_dict': netE.state_dict(),
                         'netG_state_dict': netG.state_dict(),
                         'netD_state_dict': netD.state_dict(),
                         'netDl_state_dict': netDl.state_dict(),
                         'optD_state_dict': optimizerD.state_dict(),
                         'optDl_state_dict': optimizerDl.state_dict(),
                         'optG_state_dict': optimizerG.state_dict(),
                         'optE_state_dict': optimizerE.state_dict()}

            torch.save(save_dict, os.path.join(save_dir, 'checkpoint.pkl'))
            torch.save(netE, os.path.join(save_dir, 'netE.pt'))
            torch.save(netG, os.path.join(save_dir, 'netG.pt'))


if __name__ == '__main__':
    main()
