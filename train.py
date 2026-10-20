import argparse
import os
import torch
from train_helper_ALTGVT import Trainer
from utils.dora_options import resolve_training_options


def add_training_arguments(parser):
    # DoRA settings live here so experiments can be configured directly in train.py.
    parser.add_argument('--stage', choices=['baseline', 'dora-day', 'dora-night'],
                        default='baseline', help='select the training stage')
    parser.add_argument('--baseline-checkpoint', default='',
                        help='trained CCTrans baseline .pth or .tar')
    parser.add_argument('--save-dir', default='ckpts')

    # Day adapter: edit these defaults independently.
    parser.add_argument('--dora-day-rank', type=int, default=4,
                        help='low-rank dimension')
    parser.add_argument('--dora-day-alpha', type=float, default=4.0,
                        help='DoRA scaling coefficient')
    parser.add_argument('--dora-day-lr', type=float, default=2e-4,
                        help='adapter learning rate')
    parser.add_argument('--dora-day-weight-decay', type=float, default=1e-5,
                        help='adapter weight decay')
    parser.add_argument('--dora-day-epochs', type=int, default=1200,
                        help='total adapter training epochs')
    parser.add_argument('--dora-day-val-epoch', type=int, default=1,
                        help='validation interval')
    parser.add_argument('--dora-day-val-start', type=int, default=200,
                        help='first eligible validation epoch, zero-based')
    parser.add_argument('--dora-day-train-dir', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/night/train',
                        help='training split directory')
    parser.add_argument('--dora-day-val-dir', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/night/val',
                        help='validation split directory')
    parser.add_argument('--dora-day-checkpoint', default='',
                        help='adapter weights for fresh fine-tuning without optimizer state')

    # Night adapter: edit these defaults independently.
    parser.add_argument('--dora-night-rank', type=int, default=4,
                        help='low-rank dimension')
    parser.add_argument('--dora-night-alpha', type=float, default=4.0,
                        help='DoRA scaling coefficient')
    parser.add_argument('--dora-night-lr', type=float, default=2e-4,
                        help='adapter learning rate')
    parser.add_argument('--dora-night-weight-decay', type=float, default=1e-5,
                        help='adapter weight decay')
    parser.add_argument('--dora-night-epochs', type=int, default=1200,
                        help='total adapter training epochs')
    parser.add_argument('--dora-night-val-epoch', type=int, default=1,
                        help='validation interval')
    parser.add_argument('--dora-night-val-start', type=int, default=200,
                        help='first eligible validation epoch, zero-based')
    parser.add_argument('--dora-night-train-dir', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/night/train',
                        help='training split directory')
    parser.add_argument('--dora-night-val-dir', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/night/val',
                        help='validation split directory')
    parser.add_argument('--dora-night-checkpoint', default='',
                        help='adapter weights for fresh fine-tuning without optimizer state')


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data-dir', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/day', help='data path')
    parser.add_argument('--dataset', default='sha', help='dataset name: qnrf, nwpu, sha, shb, custom')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='', type=str,
                        help='resume from a latest .ckpt (legacy .tar also supported)')
    parser.add_argument('--max-epoch', type=int, default=1200,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=200,
                        help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='the num of training process')
    parser.add_argument('--crop-size', type=int, default= 256,
                        help='the crop size of the train image')
    parser.add_argument('--wot', type=float, default=0.1, help='weight on OT loss')
    parser.add_argument('--wtv', type=float, default=0.01, help='weight on TV loss')
    parser.add_argument('--density-loss', choices=['tv', 'l2'], default='tv',
                        help='density supervision: discrete TV or adaptive Gaussian MSE')
    parser.add_argument('--wl2', type=float, default=1.0, help='weight on Gaussian L2 loss')
    parser.add_argument('--gaussian-beta', type=float, default=0.3,
                        help='sigma multiplier for mean distance to up to 3 neighbors')
    parser.add_argument('--gaussian-sigma', type=float, default=15.0,
                        help='fallback sigma in image pixels for a single head')
    parser.add_argument('--reg', type=float, default=10.0,
                        help='entropy regularization in sinkhorn')
    parser.add_argument('--num-of-iter-in-ot', type=int, default=100,
                        help='sinkhorn iterations')
    parser.add_argument('--norm-cood', type=int, default=0, help='whether to norm cood when computing distance')

    parser.add_argument('--run-name', default='CCTrans', help='run name for wandb interface/logging')
    parser.add_argument('--wandb', default=0, type=int, help='boolean to set wandb logging')
    
    add_training_arguments(parser)
    args = parser.parse_args()
    if args.gaussian_beta <= 0 or args.gaussian_sigma <= 0:
        parser.error('Gaussian beta and sigma must be positive')
    if min(args.wot, args.wtv, args.wl2) < 0:
        parser.error('Loss weights must be nonnegative')

    if args.dataset.lower() == 'qnrf':
        args.crop_size = 512
    elif args.dataset.lower() == 'nwpu':
        args.crop_size = 384
        args.val_epoch = 50
    elif args.dataset.lower() == 'sha':
        args.crop_size = 256
    elif args.dataset.lower() == 'shb':
        args.crop_size = 512
    elif args.dataset.lower() == 'custom':
        args.crop_size = 256
    else:
        raise NotImplementedError
    resolve_training_options(args, parser)
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()
