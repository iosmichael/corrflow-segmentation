import argparse
import os, time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import numpy as np
import yaml

import functional.feeder.dataset.Davis2017 as D
# dataloader with images and annotations 
import functional.feeder.dataset.DavisLoader as DL

from models.corrflow import CorrFlow
from loss import AverageMeter, compute_ll, compute_ls

import logger

config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
model_config = config['model']

def main():
    if not os.path.isdir(config['savepath']):
        os.makedirs(config['savepath'])

    log = logger.setup_logger(config['savepath'] + '/training.log')

    TrainData = D.dataloader_train(config['datapath'])
    TrainImgLoader = torch.utils.data.DataLoader(
        DL.DAVISColorizationDataset(config['datapath'], TrainData, True),
        batch_size=model_config['bsize'], shuffle=True, num_workers=model_config['num_workers'],drop_last=True
    )

    model = CorrFlow({})
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'], betas=(0.9,0.999))

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if config['resume']:
        if os.path.isfile(config['resume']):
            log.info("=> loading checkpoint '{}'".format(config['resume']))
            checkpoint = torch.load(config['resume'])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}'".format(config['resume']))
        else:
            log.info("=> No checkpoint found at '{}'".format(config['resume']))
            log.info("=> Will start from scratch.")
    else:
        log.info('=> No checkpoint file. Start from scratch.')

    start_full_time = time.time()

    for epoch in range(config['epochs']):
        log.info('This is {}-th epoch'.format(epoch))
        train(TrainImgLoader, model, optimizer, log, epoch)

    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))

def train(dataloader, model, optimizer, log, epoch):
    _loss = AverageMeter()
    n_b = len(dataloader)

    for b_i, (images_rgb, images_quantized) in enumerate(dataloader):
        model.train()
        b_s = time.perf_counter()
        adjust_lr(optimizer, epoch, b_i, n_b, model_config['lr'])

        images_rgb = [r.cuda() for r in images_rgb]
        images_quantized = [q.cuda() for q in images_quantized]
        model.module.dropout2d(images_rgb)

        optimizer.zero_grad()

        l_sim = compute_ls(model, images_rgb, images_quantized, b_i, epoch, n_b)
        l_long = compute_ll(model, images_rgb, images_quantized)

        sum_loss = l_sim + l_long * 0.1
        sum_loss.backward()
        optimizer.step()
        _loss.update(sum_loss.item())

        info = 'Loss = {:.3f}({:.3f})'.format(_loss.val, _loss.avg)
        b_t = time.perf_counter() - b_s
        for param_group in optimizer.param_groups:
            lr_now = param_group['lr']
        log.info('Epoch{} [{}/{}] {} T={:.2f}  LR={:.6f}'.format(
            epoch, b_i, n_b, info, b_t, lr_now))

        if b_i > 0 and (b_i * model_config['bsize']) % model_config['checkpoint_bsize'] < model_config['bsize']:
            log.info("Saving new checkpoint.")
            savefilename = config['savepath'] + '/checkpoint.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, savefilename)

if __name__ == '__main__':
    main()