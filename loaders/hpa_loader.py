import os
import cv2
import torch
import pandas as pd
import numpy as np

from torch.utils import data

from misc.utils import recursive_glob


class hpaLoader(data.Dataset):
    def __init__(self, root, split="train", is_transform=True,
                 img_size=(512, 512), augmentations=None,
                 no_gt=False, use_external=False, fold_num=0, num_folds=1, seed=1234):

        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.no_gt = no_gt
        self.n_classes = 28
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean_gbry = [0.0526, 0.0547, 0.0804, 0.0827]
        self.std_gbry = [0.1122, 0.1560, 0.1496, 0.1497]
        self.files = {}

        #"""
        root_img_path = os.path.join(self.root, self.split.replace('val', 'train'))
        fs = recursive_glob(rootdir=root_img_path, suffix='.png')
        uni_fs = np.unique(np.array([os.path.join(self.split.replace('val', 'train'), '_'.join(os.path.basename(f).split('_')[:-1])) for f in fs]))
        self.files[split] = uni_fs.tolist()
        if self.split != 'test' and use_external:
            root_img_path = os.path.join(self.root, 'external')
            fs = recursive_glob(rootdir=root_img_path, suffix='.png')
            uni_fs = np.unique(np.array([os.path.join('external', '_'.join(os.path.basename(f).split('_')[:-1])) for f in fs]))
            self.files[split] = self.files[split] + uni_fs.tolist()
        #"""

        list_root = 'data_list'
        if not os.path.exists(list_root):
             os.mkdir(list_root)

        list_filename = os.path.join(list_root, 'list_{}_{}-{}'.format(self.split, fold_num, num_folds)) if self.split != 'test' else os.path.join(list_root, 'list_{}'.format(self.split))
        if not os.path.exists(list_filename):
            N = len(self.files[split])
            if self.split == 'test':
                with open(list_filename, 'w') as f_test:
                    for i in range(N):
                        f_test.write(self.files[split][i] + '\n')
            else:
                torch.manual_seed(seed)
                rp = torch.randperm(N).tolist()
                start_idx = N * fold_num // num_folds
                end_idx = N * (fold_num + 1) // num_folds
                print('{:5s}: {:2d}/{:2d} [{:6d}, {:6d}] - {:6d}'.format(self.split, fold_num, num_folds, start_idx, end_idx, N))
                f_train = open(list_filename.replace('val', 'train'), 'w')
                f_val = open(list_filename.replace('train', 'val'), 'w')
                for i in range(N):
                    if i >= start_idx and i < end_idx:
                        f_val.write(self.files[split][rp[i]] + '\n')
                    else:
                        f_train.write(self.files[split][rp[i]] + '\n')
                f_train.close()
                f_val.close()
        else:
            with open(list_filename, 'r') as f:
                self.files[split] = f.read().splitlines()

        train_df = pd.read_csv(os.path.join(self.root, 'train.csv'), index_col=0)
        self.train_labels = train_df.to_dict('index')
        self.class_num_samples_train = torch.zeros(self.n_classes, dtype=torch.float, device=torch.device('cuda'))
        for i, img_id in enumerate(self.train_labels):
            lbl_str = self.train_labels[img_id]['Target']
            for l in lbl_str.split():
                self.class_num_samples_train[int(l)] += 1
        if use_external:
            external_df = pd.read_csv(os.path.join(self.root, 'HPAv18RGBY_wodpl.csv' if self.split == 'test' else 'HPAv18RGBY_WithoutUncertain_wodpl.csv'), index_col=0)
            external_labels = external_df.to_dict('index')
            class_num_samples_external = torch.zeros(self.n_classes, dtype=torch.float, device=torch.device('cuda'))
            for i, img_id in enumerate(external_labels):
                lbl_str = external_labels[img_id]['Target']
                for l in lbl_str.split():
                    class_num_samples_external[int(l)] += 1
            self.train_labels.update(external_labels)

        self.class_num_samples = self.class_num_samples_train + class_num_samples_external
        self.loss_weights = ((self.class_num_samples.sum() - self.class_num_samples) / self.class_num_samples).log()

        self.class_names = ['Nucleoplasm', 'Nuclear membrane', 'Nucleoli', 'Nucleoli fibrillar center', 'Nuclear speckles',
                            'Nuclear bodies', 'Endoplasmic reticulum', 'Golgi apparatus', 'Peroxisomes', 'Endosomes',
                            'Lysosomes', 'Intermediate filaments', 'Actin filaments', 'Focal adhesion sites', 'Microtubules',
                            'Microtubule ends', 'Cytokinetic bridge', 'Mitotic spindle', 'Microtubule organizing center', 'Centrosome',
                            'Lipid droplets', 'Plasma membrane', 'Cell junctions', 'Mitochondria', 'Aggresome',
                            'Cytosol', 'Cytoplasmic bodies', 'Rods & rings',]

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.root))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.files[self.split][index].rstrip())
        img_id = os.path.basename(img_path)

        img = np.zeros((512, 512, 4), dtype=np.uint8)
        for i, c in enumerate(['green', 'blue', 'red', 'yellow']):
            img_c = cv2.imread('{}_{}.png'.format(img_path, c), cv2.IMREAD_GRAYSCALE)
            img[:, :, i] = np.array(img_c, dtype=np.uint8)

        if not self.no_gt:
            if self.augmentations is not None:
                img = np.array(self.augmentations(img), dtype=np.uint8)
            lbl_str = self.train_labels[img_id]['Target']
            lbl = np.zeros((self.n_classes,), dtype=np.int32) # one-hot encoding
            for l in lbl_str.split():
                lbl[int(l)] = 1
        else:
            lbl = np.zeros((self.n_classes,), dtype=np.int32) # one-hot encoding

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl, [img_id]

    def transform(self, img, lbl):
        if img.shape[0] != self.img_size[0] or img.shape[1] != self.img_size[1]:
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR) # cv2.resize shape: (W, H)

        img = img.astype(np.float64) / 255.0 # Rescale images from [0, 255] to [0, 1]
        img = (img - self.mean_gbry) / self.std_gbry

        if len(img.shape) == 3:
            img = img.transpose(2, 0, 1) # NHWC -> NCHW
        else:
            img = np.expand_dims(img, axis=0) # gray-scale

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        return img, lbl



if __name__ == '__main__':
    from tqdm import tqdm
    local_path = '../datasets/HPA/'

    num_folds = 5
    for fold_num in range(num_folds):
        dst = hpaLoader(local_path, split="train", fold_num=fold_num, num_folds=num_folds, use_external=True)
    dst = hpaLoader(local_path, split="test", fold_num=0, num_folds=num_folds, use_external=True)
