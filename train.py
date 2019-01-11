import sys, os
import cv2
import torch
import argparse
import timeit
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils import data
from sklearn.metrics import f1_score
from tqdm import tqdm

from models import get_model
from loaders import get_loader, get_data_path
from misc.losses import f_beta_loss
from misc.utils import convert_state_dict, poly_lr_scheduler, AverageMeter

torch.backends.cudnn.benchmark = True

def train(args):
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    # Setup Augmentations
    data_aug = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.0/0.9)),
                ])

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, split='train', img_size=(args.img_rows, args.img_cols), augmentations=data_aug, fold_num=args.fold_num, num_folds=args.num_folds, seed=args.seed, use_external=args.use_external)
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols), fold_num=args.fold_num, num_folds=args.num_folds, seed=args.seed, use_external=args.use_external)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=2, pin_memory=True, shuffle=True, drop_last=True)
    valloader = data.DataLoader(v_loader, batch_size=1, num_workers=2, pin_memory=True)

    # Setup Model
    model = get_model(args.arch, n_classes, use_cbam=args.use_cbam)
    model.cuda()

    # Check if model has custom optimizer / loss
    if hasattr(model, 'optimizer'):
        optimizer = model.optimizer
    else:
        milestones = [5, 10, 15]
        gamma = 0.2

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.l_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        ##optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.l_rate, weight_decay=args.weight_decay)
        if args.num_cycles > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch*len(trainloader)//args.num_cycles, eta_min=args.l_rate*(gamma**len(milestones)))
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    if hasattr(model, 'loss'):
        print('Using custom loss')
        loss_fn = model.loss
    else:
        loss_fn = F.binary_cross_entropy_with_logits

    start_epoch = 0
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            model_dict = model.state_dict()
            if checkpoint.get('model_state', None) is not None:
                model_dict.update(convert_state_dict(checkpoint['model_state'], load_classifier=args.load_classifier))
            else:
                model_dict.update(convert_state_dict(checkpoint, load_classifier=args.load_classifier))

            if checkpoint.get('f1_score', None) is not None:
                start_epoch = checkpoint['epoch']
                print("Loaded checkpoint '{}' (epoch {}, f1_score {:.5f})"
                      .format(args.resume, checkpoint['epoch'], checkpoint['f1_score']))
            elif checkpoint.get('epoch', None) is not None:
                start_epoch = checkpoint['epoch']
                print("Loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))

            model.load_state_dict(model_dict)

            if checkpoint.get('optimizer_state', None) is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 

    for epoch in range(start_epoch, args.n_epoch):
        start_train_time = timeit.default_timer()

        if args.num_cycles == 0:
            scheduler.step(epoch)

        recon_scale = 4
        bce_loss_sum = 0.0
        cooc_loss_sum = 0.0
        mse_loss_sum = 0.0
        model.train()
        optimizer.zero_grad()
        for i, (images, labels, _) in enumerate(trainloader):
            if args.num_cycles > 0 and (i+1) % args.iter_size == 0:
                iter_num = i + epoch * len(trainloader)
                scheduler.step(iter_num % (args.n_epoch * len(trainloader) // args.num_cycles)) # Cosine Annealing with Restarts

            images = images.cuda()
            labels = labels.cuda()

            images_ref = images.clone() # for image (4 channels) reconstruction
            if recon_scale != 4:
                images_ref = F.interpolate(images_ref, scale_factor=recon_scale/4., mode='bilinear', align_corners=False)

            sum_labels = labels.sum(1).long()
            sum_labels = torch.where(sum_labels <= 4, sum_labels, torch.zeros_like(sum_labels))

            outputs, outputs_cooc, recons = model(images, recon_scale=recon_scale)

            bce_loss = loss_fn(outputs[:labels.size(0), :], labels, pos_weight=t_loader.loss_weights)
            bce_loss = bce_loss / float(args.iter_size)
            bce_loss_sum = bce_loss_sum + bce_loss

            cooc_loss = F.cross_entropy(outputs_cooc[:labels.size(0), :], sum_labels)
            cooc_loss = cooc_loss / float(args.iter_size)
            cooc_loss = args.lambda_cooc_loss * cooc_loss
            cooc_loss_sum = cooc_loss_sum + cooc_loss

            mse_loss = F.mse_loss(recons, images_ref)
            mse_loss = mse_loss / float(args.iter_size)
            mse_loss = args.lambda_mse_loss * mse_loss
            mse_loss_sum = mse_loss_sum + mse_loss

            loss = bce_loss + cooc_loss + mse_loss
            loss.backward()

            if (i+1) % args.print_train_freq == 0:
                print("Epoch [%3d/%3d] Iter [%6d/%6d] Loss: BCE %.4f / COOC %.4f / MSE %.4f" % (epoch+1, args.n_epoch, i+1, len(trainloader), bce_loss_sum, cooc_loss_sum, mse_loss_sum))

            if (i+1) % args.iter_size == 0 or i == len(trainloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                bce_loss_sum = 0.0
                cooc_loss_sum = 0.0
                mse_loss_sum = 0.0

        state = {'epoch': epoch+1,
                 'model_state': model.state_dict(),}
                 #'optimizer_state': optimizer.state_dict(),}
        torch.save(state, "checkpoints/{}_{}_{}_{}x{}_{}-{}_model.pth".format(args.arch, args.dataset, epoch+1, args.img_rows, args.img_cols, args.fold_num, args.num_folds))

        if (epoch+1) % args.eval_freq == 0:
            weak_samples = 0
            thresh = 0.5
            y_true = np.zeros((v_loader.__len__(), n_classes), dtype=np.int32)
            y_pred = np.zeros((v_loader.__len__(), n_classes), dtype=np.int32)
            mean_loss_val = AverageMeter()
            model.eval()
            with torch.no_grad():
                for i_val, (images_val, labels_val, _) in tqdm(enumerate(valloader)):
                    images_val = images_val.cuda()
                    labels_val = labels_val.cuda()

                    outputs_val = model(images_val)
                    prob = F.sigmoid(outputs_val)
                    max_pred = prob.max(1)[1].cpu().numpy()
                    pred = (prob >= thresh)
                    pred_sum = pred.sum(1)

                    bce_loss_val = loss_fn(outputs_val, labels_val, pos_weight=v_loader.loss_weights)
                    loss_val = bce_loss_val
                    mean_loss_val.update(loss_val, n=images_val.size(0))

                    y_true[i_val, :] = labels_val.long().cpu().numpy()
                    y_pred[i_val, :] = pred.long().cpu().numpy()

                    for k in range(images_val.size(0)):
                        if pred_sum[k] == 0:
                            y_pred[i_val, max_pred[k]] = 1
                            weak_samples += 1

            f1_score_val = f1_score(y_true, y_pred, labels=[l for l in range(n_classes)], average='macro')
            print('F1-score (macro): {:.5f}'.format(f1_score_val))
            print('Mean val loss: {:.4f}'.format(mean_loss_val.avg))
            state['f1_score'] = f1_score_val
            mean_loss_val.reset()

            for k in range(n_classes):
                num = y_pred[:, k].sum()
                print('{:2d}: {:5d} ({:.5f}) | {:5d} ({:.5f})'.format(k, num, float(num)/y_pred.sum(), v_loader.class_num_samples[k].long(), v_loader.class_num_samples[k]/v_loader.class_num_samples.sum()))
            print('# of weak samples: {}'.format(weak_samples))

        torch.save(state, "checkpoints/{}_{}_{}_{}x{}_{}-{}_model.pth".format(args.arch, args.dataset, epoch+1, args.img_rows, args.img_cols, args.fold_num, args.num_folds))

        elapsed_train_time = timeit.default_timer() - start_train_time
        print('Training time (epoch {0:5d}): {1:10.5f} seconds'.format(epoch+1, elapsed_train_time))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet18', 
                        help='Architecture to use [\'fcn8s, unet, segnet, pspnet, icnet, etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='HPA', 
                        help='Dataset to use [\'pascal, camvid, ade20k, cityscapes, etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=512, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=512, 
                        help='Width of the input image')

    parser.add_argument('--n_epoch', nargs='?', type=int, default=20, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=40, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-2, 
                        help='Learning Rate')
    parser.add_argument('--momentum', nargs='?', type=float, default=0.9, 
                        help='Momentum')
    parser.add_argument('--weight_decay', nargs='?', type=float, default=1e-4, 
                        help='Weight Decay')
    parser.add_argument('--iter_size', nargs='?', type=int, default=1,
                        help='Accumulated batch gradient size')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')

    parser.add_argument('--use_external', dest='use_external', action='store_true',
                        help='Enable to load external data | True by default')
    parser.add_argument('--no-use_external', dest='use_external', action='store_false',
                        help='Disable to load external data | True by default')
    parser.set_defaults(use_external=True)

    parser.add_argument('--load_classifier', dest='load_classifier', action='store_true',
                        help='Enable to load pretrained classifier weights | True by default')
    parser.add_argument('--no-load_classifier', dest='load_classifier', action='store_false',
                        help='Disable to load pretrained classifier weights | True by default')
    parser.set_defaults(load_classifier=True)

    parser.add_argument('--use_cbam', dest='use_cbam', action='store_true',
                        help='Enable to use CBAM | False by default')
    parser.add_argument('--no-use_cbam', dest='use_cbam', action='store_false',
                        help='Disable to use CBAM | False by default')
    parser.set_defaults(use_cbam=False)

    parser.add_argument('--seed', nargs='?', type=int, default=1234, 
                        help='Random seed')
    parser.add_argument('--num_cycles', nargs='?', type=int, default=1, 
                        help='Cosine Annealing Cyclic LR')

    parser.add_argument('--fold_num', nargs='?', type=int, default=0,
                        help='Fold number in each class for training')
    parser.add_argument('--num_folds', nargs='?', type=int, default=5,
                        help='Number of folds for training')
    parser.add_argument('--print_train_freq', nargs='?', type=int, default=20,
                        help='Frequency (iterations) of training logs display')
    parser.add_argument('--eval_freq', nargs='?', type=int, default=5,
                        help='Frequency (epochs) of evaluation of current model')

    parser.add_argument('--lambda_cooc_loss', nargs='?', type=float, default=0.1, 
                        help='lambda_cooc_loss')
    parser.add_argument('--lambda_mse_loss', nargs='?', type=float, default=0.1, 
                        help='lambda_mse_loss')

    args = parser.parse_args()
    print(args)
    train(args)
