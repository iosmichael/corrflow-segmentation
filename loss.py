import torch
import numpy as np
import torch.functional as F

def compute_ls(model, image_rgb, image_q, bi, epoch, n_b):
    eps_s, eps_e = 0.9, 0.6
    b, c, h, w = image_rgb[0].size()

    # Loss similarity
    l_sim = 0

    for i in range(2): # 3-1
        ref_g = image_rgb[i]
        tar_g = image_rgb[i + 1]
        tar_c = image_q[i + 1]
        tar_c = torch.squeeze(tar_c, 1)

        total_batch = args.epochs * n_b
        current_batch = epoch * n_b + bi
        thres = eps_s - (eps_s - eps_e) * current_batch / total_batch
        truth = np.random.random() < thres
        ref_c = image_q[i] if truth or (i == 0) else outputs

        outputs = model(ref_g, ref_c, tar_g)
        outputs = F.interpolate(outputs, (h, w), mode='bilinear')

        loss = cross_entropy(outputs, tar_c, size_average=True)
        l_sim += loss

    return l_sim

def compute_ll(model, image_rgb, image_q):
    b, c, h, w = image_rgb[0].size()
    # Loss long
    l_long = 0

    for i in range(1,3):
        for j in range(i):
            ref_g = image_rgb[j]
            if j == 0:
                ref_c = image_q[j]
            else:
                ref_c = outputs

            tar_g = image_rgb[j + 1]

            outputs = model(ref_g, ref_c, tar_g)
            outputs = F.interpolate(outputs, (h, w), mode='bilinear')

        for j in range(i,0,-1):
            ref_g = image_rgb[j]
            ref_c = outputs
            tar_g = image_rgb[j-1]
            outputs = model(ref_g, ref_c, tar_g)
            outputs = F.interpolate(outputs, (h, w), mode='bilinear')
        tar_c = image_q[0]
        tar_c = torch.squeeze(tar_c, 1)

        loss = cross_entropy(outputs, tar_c, size_average=True)
        l_long += loss

    return l_long


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_lr(optimizer, epoch, batch, n_b, lr_init):
    iteration = (batch + epoch * n_b) * args.bsize

    if iteration <= 400000:
        lr = lr_init
    elif iteration <= 600000:
        lr = lr_init * 0.5
    elif iteration <= 800000:
        lr = lr_init * 0.25
    elif iteration <= 1000000:
        lr = lr_init * 0.125
    else:
        lr = lr_init * 0.0625

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduction='mean'):
    if size_average:
        reduction = 'mean'
    return F.nll_loss(torch.log(input + 1e-8), target, weight, None, ignore_index, None, reduction)