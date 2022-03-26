import os
import pdb
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = (os.path.split(curPath)[0])
sys.path.append(rootPath)

import argparse
import logging
import random
import re
import time

import cv2

import torch.nn as nn

from src.config import *
from miou_utils import compute_iu, fast_cm
from src.util import *
from tensorboardX import SummaryWriter
from utils.sync_batchnorm.replicate import patch_replication_callback

def get_arguments():
    parser = argparse.ArgumentParser(description="Full Pipeline Training")
    parser.add_argument("--train-dir", type=str, default=TRAIN_DIR,
                        help="Path to the training set directory.")
    parser.add_argument("--val-dir", type=str, default=VAL_DIR,
                        help="Path to the validation set directory.")
    parser.add_argument("--train-list", type=str, nargs='+', default=TRAIN_LIST,
                        help="Path to the training set list.")
    parser.add_argument("--val-list", type=str, nargs='+', default=VAL_LIST,
                        help="Path to the validation set list.")
    parser.add_argument("--shorter-side", type=int, nargs='+', default=SHORTER_SIDE,
                        help="Shorter side transformation.")
    parser.add_argument("--crop-size", type=int, nargs='+', default=CROP_SIZE,
                        help="Crop size for training,")
    parser.add_argument("--normalise-params", type=list, default=NORMALISE_PARAMS,
                        help="Normalisation parameters [scale, mean, std],")
    parser.add_argument("--batch-size", type=int, nargs='+', default=BATCH_SIZE,
                        help="Batch size to train the segmenter model.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="Number of workers for pytorch's dataloader.")
    parser.add_argument("--num-classes", type=int, nargs='+', default=NUM_CLASSES,
                        help="Number of output classes for each task.")
    parser.add_argument("--low-scale", type=float, nargs='+', default=LOW_SCALE,
                        help="Lower bound for random scale")
    parser.add_argument("--high-scale", type=float, nargs='+', default=HIGH_SCALE,
                        help="Upper bound for random scale")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="Label to ignore during training")

    # Framework
    parser.add_argument("--enc-pretrained", type=bool, default=ENC_PRETRAINED,
                        help='Whether to init with imagenet weights.')
    # General
    parser.add_argument("--evaluate", type=bool, default=EVALUATE,
                        help='If true, only validate segmentation.')
    parser.add_argument("--freeze-bn", type=bool, nargs='+', default=FREEZE_BN,
                        help='Whether to keep batch norm statistics intact.')
    parser.add_argument("--num-segm-epochs", type=int, nargs='+', default=NUM_SEGM_EPOCHS,
                        help='Number of epochs to train for segmentation network.')
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY,
                        help='Print information every often.')
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help='Seed to provide (near-)reproducibility.')
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Path to directory for storing checkpoints.")
    parser.add_argument("--ckpt-path", type=str, default=CKPT_PATH,
                        help="Path to the checkpoint file.")

    # Optimisers
    parser.add_argument("--lr-enc", type=float, nargs='+', default=LR_ENC,
                        help="Learning rate for encoder.")
    parser.add_argument("--lr-dec", type=float, nargs='+', default=LR_DEC,
                        help="Learning rate for decoder.")
    parser.add_argument("--mom-enc", type=float, nargs='+', default=MOM_ENC,
                        help="Momentum for encoder.")
    parser.add_argument("--mom-dec", type=float, nargs='+', default=MOM_DEC,
                        help="Momentum for decoder.")
    parser.add_argument("--wd-enc", type=float, nargs='+', default=WD_ENC,
                        help="Weight decay for encoder.")
    parser.add_argument("--wd-dec", type=float, nargs='+', default=WD_DEC,
                        help="Weight decay for decoder.")
    parser.add_argument("--optim-dec", type=str, default=OPTIM_DEC,
                        help="Optimiser algorithm for decoder.")

    #tensorboardX
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="tensorboarX log.")
    return parser.parse_args()

def create_segmenter(num_classes):

    from models.SGCPNet import SGCPNet
    return SGCPNet(num_classes)


def create_loaders(
    train_dir, val_dir, train_list, val_list,
    shorter_side, crop_size, low_scale, high_scale,
    normalise_params, batch_size, num_workers, ignore_label
    ):
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from src.datasets import CityscapesDataset as Dataset
    from src.datasets import Pad, RandomCrop, RandomMirror, ResizeShorterScale, ToTensor, Normalise, ValScale

    composed_trn = transforms.Compose([ResizeShorterScale(shorter_side, low_scale, high_scale),
                                                   Pad(crop_size, [123.675, 116.28 , 103.53], ignore_label),
                                                   RandomMirror(),
                                                   RandomCrop(crop_size),
                                                   Normalise(*normalise_params),
                                                   ToTensor()])

    composed_val = transforms.Compose([Normalise(*normalise_params), ToTensor()])


    trainset = Dataset(data_file=train_list,
                                   data_dir=train_dir,
                                   transform_trn=composed_trn,
                                   transform_val=composed_val)

    valset = Dataset(data_file=val_list,
                                 data_dir=val_dir,
                                 transform_trn=None,
                                 transform_val=composed_val)

    logger.info(" Created train set = {} examples, val set = {} examples".format(len(trainset), len(valset)))

    train_loader = DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              drop_last=True)

    val_loader = DataLoader(valset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=True)

    return train_loader, val_loader

def create_optimisers(lr_enc, lr_dec, mom_enc, mom_dec, wd_enc, wd_dec, param_enc, param_dec, optim_dec):
    optim_enc = torch.optim.SGD(param_enc, lr=lr_enc, momentum=mom_enc, weight_decay=wd_enc)
    if optim_dec == 'sgd':
        optim_dec = torch.optim.SGD(param_dec, lr=lr_dec, momentum=mom_dec, weight_decay=wd_dec)
    elif optim_dec == 'adam':
        optim_dec = torch.optim.Adam(param_dec, lr=lr_dec, weight_decay=wd_dec, eps=1e-3)
    return optim_enc, optim_dec

def load_ckpt(ckpt_path, ckpt_dict):
    best_val = epoch_start = 0
    if os.path.exists(args.ckpt_path):
        ckpt = torch.load(ckpt_path)
        for (k, v) in ckpt_dict.items():
            if k in ckpt:
                v.load_state_dict(ckpt[k])
        best_val = ckpt.get('best_val', 0)
        epoch_start = ckpt.get('epoch_start', 0)
        logger.info(" Found checkpoint at {} with best_val {:.4f} at epoch {}".format(ckpt_path, best_val, epoch_start))
    return best_val, epoch_start

def train_segmenter(segmenter, train_loader, optim_enc, optim_dec, epoch, total_epoch, segm_crit, freeze_bn, lr_enc, lr_dec, tblog):
    train_loader.dataset.set_stage('train')
    segmenter.train()
    if freeze_bn:
        for m in segmenter.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    max_itr = total_epoch*len(train_loader)

    for i, sample in enumerate(train_loader):
        now_itr = epoch*len(train_loader) + i
        now_lr_enc = lr_enc * (1 - now_itr/(max_itr + 1))**0.9
        now_lr_dec = lr_dec * (1 - now_itr/(max_itr + 1))**0.9
        optim_enc.param_groups[0]['lr'] = now_lr_enc
        optim_dec.param_groups[0]['lr'] = now_lr_dec

        start = time.time()
        input = sample['image'].cuda()
        target = sample['mask'].cuda().to(1)
        input_var = torch.autograd.Variable(input).float()
        target_var = torch.autograd.Variable(target).long()
        output = segmenter(input_var).to(1)

        output = nn.functional.interpolate(output, size=target_var.size()[1:], mode='bilinear', align_corners=False)
        soft_output = nn.LogSoftmax()(output)

        # Compute loss and backpropagate
        loss = segm_crit(soft_output, target_var)
        optim_enc.zero_grad()
        optim_dec.zero_grad()
        loss.backward()
        optim_enc.step()
        optim_dec.step()
        losses.update(loss.item())
        batch_time.update(time.time() - start)
        tblog.add_scalar("loss", losses.avg, now_itr)

    logger.info(' Train epoch: {}|{}\t' 'Avg.Loss: {:.3f}\t' 'Avg.Time: {:.3f}'.format(epoch, total_epoch, losses.avg, batch_time.avg))
def validate(segmenter, val_loader, epoch, num_classes=-1):

    val_loader.dataset.set_stage('val')
    segmenter.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            start = time.time()
            input = sample['image']
            target = sample['mask']

            input_var = torch.autograd.Variable(input).float().cuda()
            # Compute output
            output = segmenter(input_var)
            output = cv2.resize(output[0,:num_classes].data.cpu().numpy().transpose(1, 2, 0),
                                target.size()[1:][::-1],
                                interpolation=cv2.INTER_CUBIC).argmax(axis=2).astype(np.uint8)
            # Compute IoU
            gt = target[0].data.cpu().numpy().astype(np.uint8)
            gt_idx = gt < num_classes # Ignore every class index larger than the number of classes
            cm += fast_cm(output[gt_idx], gt[gt_idx], num_classes)

    ious = compute_iu(cm)*100.
    logger.info(" IoUs: {}".format(ious))
    miou = np.mean(ious)
    logger.info(' Val epoch: {}\tMean IoU: {:.3f}'.format(epoch, miou))
    return miou

def main():
    global args, logger, tblog

    args = get_arguments()
    logger = logging.getLogger(__name__)
    tblog = SummaryWriter(args.log_dir)

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    device = torch.device(0)
    segmenter = nn.DataParallel(create_segmenter(args.num_classes))
    patch_replication_callback(segmenter)
    segmenter.to(device)
    logger.info(" Loaded Segmenter, #PARAMS={:3.2f}M".format(compute_params(segmenter) / 1e6))
    best_val, epoch_strat = load_ckpt(args.ckpt_path, {'segmenter': segmenter})
    if epoch_strat !=0:
        epoch_strat = epoch_strat +1
    segm_crit = nn.NLLLoss2d(ignore_index=args.ignore_label).cuda()

    ## Saver ##
    if args.evaluate is False:
        saver = Saver(args=vars(args),
                      ckpt_dir=args.snapshot_dir,
                      best_val=best_val,
                      condition=lambda x, y: x >= y)

    logger.info(" Training Process Starts")
    start = time.time()
    torch.cuda.empty_cache()
    train_loader, val_loader = create_loaders(args.train_dir,
                                              args.val_dir,
                                              args.train_list,
                                              args.val_list,
                                              args.shorter_side,
                                              args.crop_size,
                                              args.low_scale,
                                              args.high_scale,
                                              args.normalise_params,
                                              args.batch_size,
                                              args.num_workers,
                                              args.ignore_label)
    if args.evaluate:
        return validate(segmenter, val_loader, 0, num_classes=args.num_classes)

    ## Optimisers ##
    enc_params = []
    dec_params = []
    for k,v in segmenter.named_parameters():
        if bool(re.match(".*l1.*|.*l2.*|.*l3.*|.*l4.*|.*l5", k)):
            enc_params.append(v)
        else:
            dec_params.append(v)
    optim_enc, optim_dec = create_optimisers(args.lr_enc,
                                             args.lr_dec,
                                             args.mom_enc,
                                             args.mom_dec,
                                             args.wd_enc,
                                             args.wd_dec,
                                             enc_params,
                                             dec_params,
                                             args.optim_dec)

    for epoch in range(epoch_strat,args.num_segm_epochs):

        train_segmenter(segmenter,
                                    train_loader,
                                    optim_enc,
                                    optim_dec,
                                    epoch,
                                    args.num_segm_epochs,
                                    segm_crit,
                                    args.freeze_bn,
                                    args.lr_enc,
                                    args.lr_dec,
                                    tblog)


    saver.save(100, {'segmenter': segmenter.state_dict(), 'epoch_start': epoch}, logger)
    miou = validate(segmenter, val_loader, epoch+1, args.num_classes)
    print(miou)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
