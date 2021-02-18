import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from data.dataset import ImageDataset
from models.loss import GANLoss
import models.discriminator_vgg_arch as SRGAN_arch
import models.RRDBNet_arch as RRDBNet_arch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str)
    parser.add_argument('-load', type=str, default='')
    parser.add_argument('-data', type=str, default='path')
    parser.add_argument('-nblock', type=int, default=23)
    args = parser.parse_args()

    writer = SummaryWriter(log_dir='../tensorboard')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    netG = RRDBNet_arch.RRDBNet(in_nc=3, out_nc=3,
                                nf=64, nb=args.nblock).to(device)
    netD = SRGAN_arch.NLayerDiscriminator(input_nc=3, ndf=3, n_layers=3).to(device)
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=34, use_bn=False,
                                          use_input_norm=True, device=device)
    netF.eval()
    netG.train()
    netD.train()

    optimizers = []
    schedulers = []

    cri_pix = nn.L1Loss().to(device)
    cri_fea = nn.L1Loss().to(device)
    cri_gan = GANLoss('ragan').to(device)

    optim_params = []
    for k, v in netG.named_parameters():
        if v.requires_grad:
            optim_params.append(v)

    optimizer_G = torch.optim.Adam(optim_params,
                                   lr=1e-4,
                                   weight_decay=0,
                                   betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(),
                                   lr=1e-4,
                                   weight_decay=0,
                                   betas=(0.9, 0.999))

    optimizers.append(optimizer_G)
    optimizers.append(optimizer_D)

    for optimizer in optimizers:
        schedulers.append(MultiStepLR(optimizer,
                                      milestones=[100, 500, 1000, 1500, 2000, 2500, 3000],
                                      gamma=0.5))

    train_set = ImageDataset("path")
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    current_step = 0

    for _ in range(50):
        for _, train_data in enumerate(train_loader):
            current_step += 1

            var_L = train_data['LQ'].to(device)
            var_H = train_data['GT'].to(device)
            var_ref = var_H

            for p in netD.parameters():
                p.requires_grad = False

            optimizer_G.zero_grad()
            fake_H = netG(var_L.detach())

            l_g_total = 0

            l_g_pix = 1e-2 * cri_pix(fake_H, var_H)
            l_g_total += l_g_pix

            real_fea = netF(var_H).detach()
            fake_fea = netF(fake_H)
            l_g_fea = cri_fea(fake_fea, real_fea)
            l_g_total += l_g_fea

            pred_g_fake = netD(fake_H)
            pred_d_real = netD(var_ref).detach()
            l_g_gan = 5e-3 * (
                    cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                    cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            l_g_total += l_g_gan

            l_g_total.backward()
            optimizer_G.step()

            # 이제 D
            for p in netD.parameters():
                p.requires_grad = True

            optimizer_D.zero_grad()
            pred_d_real = netD(var_ref)
            pred_d_fake = netD(fake_H.detach())  # detach to avoid BP to G
            l_d_real = cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            l_d_fake = cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            l_d_total = (l_d_real + l_d_fake) / 2

            l_d_total.backward()
            optimizer_D.step()

            for scheduler in schedulers:
                scheduler.step()

            if current_step % 100 == 0:
                writer.add_scalar('l_g_pix', l_g_pix.item())
                writer.add_scalar('l_g_fea', l_g_fea.item())
                writer.add_scalar('l_g_gan', l_g_gan.item())
                writer.add_scalar('l_d_real', l_d_real.item())
                writer.add_scalar('l_d_fake', l_d_fake.item())

                writer.add_scalar('D_real', torch.mean(pred_d_real.detach()))
                writer.add_scalar('D_fake', torch.mean(pred_d_fake.detach()))


if __name__ == '__main__':
    main()
