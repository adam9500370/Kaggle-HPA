import sys, os
import cv2
import torch
import argparse
import timeit
import random
import collections
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.backends import cudnn
from torch.utils import data
from sklearn.metrics import f1_score
from tqdm import tqdm

from models import get_model
from loaders import get_loader, get_data_path
from misc.utils import convert_state_dict, flip, AverageMeter

cudnn.benchmark = True

def test(args):
    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[:model_file_name.find('_')]

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split=args.split, is_transform=True, img_size=(args.img_rows, args.img_cols), no_gt=args.no_gt, seed=args.seed, use_external=args.use_external)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    n_classes = loader.n_classes
    testloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # Setup Model
    model = get_model(model_name, n_classes, use_cbam=args.use_cbam)
    model.cuda()

    checkpoint = torch.load(args.model_path)
    state = convert_state_dict(checkpoint['model_state'])
    model_dict = model.state_dict()
    model_dict.update(state)
    model.load_state_dict(model_dict)

    if checkpoint.get('f1_score', None) is not None:
        print("Loaded checkpoint '{}' (epoch {}, f1_score {:.5f})"
              .format(args.model_path, checkpoint['epoch'], checkpoint['f1_score']))
    else:
        print("Loaded checkpoint '{}' (epoch {})"
              .format(args.model_path, checkpoint['epoch']))

    weak_samples = 0
    y_true = np.zeros((loader.__len__(), n_classes), dtype=np.int32)
    y_pred = np.zeros((loader.__len__(), n_classes), dtype=np.int32)
    y_prob = np.zeros((loader.__len__(), n_classes), dtype=np.float32)
    y_pow = np.zeros((loader.__len__(),), dtype=np.int64)
    pow_base = 2 ** np.arange(n_classes)
    pred_dict = collections.OrderedDict()
    if args.use_leak:
        leak_df = pd.read_csv(os.path.join(data_path, 'TestEtraMatchingUnder_259_R14_G12_B10.csv'), index_col='Test')[['Extra', 'SimR', 'SimG', 'SimB']]
        leak_dict = leak_df.to_dict('index')
    model.eval()
    with torch.no_grad():
        for i, (images, labels, names) in tqdm(enumerate(testloader)):
            images = images.cuda()
            outputs = model(images)
            prob = F.sigmoid(outputs)

            if args.tta:
                sum_prob = prob
                for j in range(7):
                    images = torch.transpose(images, 2, 3) if j % 2 == 0 else flip(images, dim=3)
                    outputs = model(images)
                    prob = F.sigmoid(outputs)
                    sum_prob = sum_prob + prob
                prob = sum_prob / 8.

            if not args.no_gt:
                y_true[i*args.batch_size:i*args.batch_size+images.size(0), :] = labels.long().cpu().numpy()
            y_prob[i*args.batch_size:i*args.batch_size+images.size(0), :] = prob.cpu().numpy()
            y_pred[i*args.batch_size:i*args.batch_size+images.size(0), :] = (prob >= args.thresh).long().cpu().numpy()

            for k in range(images.size(0)):
                pred = np.where(y_pred[i*args.batch_size+k, :] == 1)[0].tolist()
                if len(pred) == 0:
                    pred = [prob.max(1)[1][k].cpu().numpy()]
                    y_pred[i*args.batch_size+k, pred] = 1
                    weak_samples += 1
                name = names[0][k]
                if args.use_leak:
                    if leak_dict.get(name, None) is not None:
                        sum_sim = leak_dict[name]['SimR'] + leak_dict[name]['SimG'] + leak_dict[name]['SimB']
                        if sum_sim <= 16:#4:
                            extra_label_name = '_'.join(leak_dict[name]['Extra'].split('_')[1:])
                            if loader.train_labels.get(extra_label_name, None) is not None:
                                pred_dict[name] = loader.train_labels[extra_label_name]['Target']

                if pred_dict.get(name, None) is None:
                    pred_dict[name] = ' '.join(map(str, pred))

            y_pow[i*args.batch_size:i*args.batch_size+images.size(0)] = (y_pred[i*args.batch_size:i*args.batch_size+images.size(0), :] * pow_base).sum(1)

    fold_num, num_folds = model_file_name.split('_')[4].split('-')
    prob_file_name = '{}_{}x{}_{}_{}_{}-{}'.format(args.split, args.img_rows, args.img_cols, model_name, checkpoint['epoch'], fold_num, num_folds)
    np.save('prob-{}.npy'.format(prob_file_name), y_prob)

    if not args.no_gt:
        f1_score_val = f1_score(y_true, y_pred, labels=[l for l in range(n_classes)], average='macro')
        print('F1-score (macro): {:.5f}'.format(f1_score_val))

    for i in range(n_classes):
        num = y_pred[:, i].sum()
        print('{:2d}: {:5d} ({:.5f}) | {:5d} ({:.5f})'.format(i, num, float(num)/y_pred.sum(), loader.class_num_samples_train[i].long(), loader.class_num_samples_train[i]/loader.class_num_samples_train.sum()))
    print('# of weak samples: {}'.format(weak_samples))

    uni, cnt = np.unique(y_pow, return_counts=True)
    sorted_idx = np.argsort(cnt)
    for i in range(len(uni)*9//10, len(uni)):
        uni_b = '{:028b}'.format(uni[sorted_idx[i]])
        cls = []
        for j in range(n_classes):
            if int(uni_b[n_classes-1-j]) == 1:
                cls.append(j)
        print('{:20s} {:5d}'.format(cls, cnt[sorted_idx[i]]))

    # Create submission
    csv_file_name = '{}_{}x{}_{}_{}_{}-{}_{}'.format(args.split, args.img_rows, args.img_cols, model_name, checkpoint['epoch'], fold_num, num_folds, args.thresh)
    csv_file_name = csv_file_name + '_leak.csv' if args.use_leak else csv_file_name + '.csv'
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['Id']
    sub.columns = ['Predicted']
    sub.to_csv(csv_file_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='resnet18_HPA_20_0-5_model.pth',
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='HPA',
                        help='Dataset to use [\'pascal, camvid, ade20k, cityscapes, etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=512,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=512,
                        help='Width of the input image')

    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='test',
                        help='Split of dataset to test on')

    parser.add_argument('--use_external', dest='use_external', action='store_true',
                        help='Enable to load external data | True by default')
    parser.add_argument('--no-use_external', dest='use_external', action='store_false',
                        help='Disable to load external data | True by default')
    parser.set_defaults(use_external=True)

    parser.add_argument('--use_cbam', dest='use_cbam', action='store_true',
                        help='Enable to use CBAM | False by default')
    parser.add_argument('--no-use_cbam', dest='use_cbam', action='store_false',
                        help='Disable to use CBAM | False by default')
    parser.set_defaults(use_cbam=False)

    parser.add_argument('--no_gt', dest='no_gt', action='store_true',
                        help='Disable verification | True by default')
    parser.add_argument('--gt', dest='no_gt', action='store_false',
                        help='Enable verification | True by default')
    parser.set_defaults(no_gt=True)

    parser.add_argument('--use_leak', dest='use_leak', action='store_true',
                        help='Enable to use data leak | False by default')
    parser.add_argument('--no-use_leak', dest='use_leak', action='store_false',
                        help='Disable to use data leak | False by default')
    parser.set_defaults(use_leak=False)

    parser.add_argument('--seed', nargs='?', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--thresh', nargs='?', type=float, default=0.5,
                        help='Threshold of prediction')

    parser.add_argument('--tta', dest='tta', action='store_true',
                        help='Enable Test Time Augmentation (TTA) with horizontal flip | True by default')
    parser.add_argument('--no-tta', dest='tta', action='store_false',
                        help='Disable Test Time Augmentation (TTA) with horizontal flip | True by default')
    parser.set_defaults(tta=True)

    args = parser.parse_args()
    print(args)
    test(args)
