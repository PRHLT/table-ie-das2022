import pytorch_lightning as pl
import torch
from torch_geometric.data import DataLoader
import multiprocessing
# from models import Collator
import random, numpy as np
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ColorJitter, ToTensor
from torchvision import  transforms as transforms_tv
from data import transforms_Graph
import time, glob, os
from data.graph_dataset_header import ABPDataset_Header as ABPDataset
conjugate = False

def worker_init_fn(_):
    # We need to reset the Numpy and Python PRNG, or we will get the
    # same numbers in each epoch (when the workers are re-generated)
    random.seed(torch.initial_seed() % 2 ** 31)
    np.random.seed(torch.initial_seed() % 2 ** 31)


class GraphDataset(pl.LightningDataModule):

    def __init__(self, opts=None, logger=None):
        super().__init__()
        # self.setup(opts)
        self.opts = opts
        self.logger = logger
        self.work_dir = opts.work_dir
        self.setup_()



    def prepare_lists(self, ):
        list_files = get_all(self.opts.data_path)
        if not self.opts.fold_paths:
            random.shuffle(list_files)
            splits = chunkIt(list_files, 4)
        if self.opts.test_lst and self.opts.train_lst:
            splits = get_trainTest(self.opts)
        else:
            if self.opts.myfolds:
                self.logger.info("Using our folds")
                splits = chunkWithMyFold(list_files, self.opts)
            else:
                splits = chunkWithFold(list_files, self.opts)
        # check_splits(splits)
        [print(len(z)) for z in splits]
        all_splits = []
        for s in splits:
            all_splits.extend(s)
        all_splits = list(set(all_splits))
        self.logger.info("A total of {} diferent files".format(len(all_splits)))
        Start_time = time.time()
        if len(splits) > 2:
            list_te, list_train, list_val = splits
            list_train_val = (list_train, list_val)
            self.logger.info(f'{len(list_te)} files for test, {len(list_train)} files for train and {len(list_val)} files for val')
        else:
            list_te, list_train_val = splits
            list_train_val = (list_train_val, )
            self.logger.info(f'{len(list_te)} files for test, {len(list_train_val[0])} files for train and no val')
        return list_te, list_train_val

    def setup_(self, ):
        self.logger.info("-----------------------------------------------")
        transforms_composed = transforms_Graph.build_transforms(self.opts)
        # transform = None
        # pre_transform = T.Compose([
        #     T.Constant(value=1),
            # T.Distance(),
            # T.KNNGraph(k=6),
        # ])
        pre_transform = None
        self.logger.info(f'Val {self.opts.do_val}')
        self.list_te, tuple_list_valtrain = self.prepare_lists()
        if len(tuple_list_valtrain) == 2:
            self.list_train, self.list_val = tuple_list_valtrain
        else:
            self.list_train = tuple_list_valtrain[0]

        self.dataset_tr = ABPDataset(root=self.opts.data_path, split="train", flist=self.list_train, transform=transforms_composed,
                                pre_transform=pre_transform, opts=self.opts)
        self.dataset_tr.process()
        
        if not self.opts.do_val:
            self.list_val = self.list_train
        
        self.dataset_dev = ABPDataset(root=self.opts.data_path, split="dev", flist=self.list_val, transform=None,
                            pre_transform=pre_transform, opts=self.opts)
        self.dataset_dev.process()
        
        self.test_dataset = ABPDataset(root=self.opts.data_path, split="test", flist=self.list_te, pre_transform=pre_transform,
                                 transform=None,
                                 opts=self.opts)
        # test_dataloader = ABPDataset_BIESO(root=opts.data_path, split="test", flist=list_tr, transform=transform, opts=opts)
        self.test_dataset.process()
        self.logger.info("A total of {} labels in test dataset".format(len(self.test_dataset.labels)))

    def train_dataloader(self):
        # trainloader_train = torch.utils.data.DataLoader(self.tr_dataset, batch_size=self.opts.batch_size, shuffle=True, num_workers=0)
        num_workers = multiprocessing.cpu_count()
        # num_workers = 1
        train_dataloader = DataLoader(
            self.dataset_tr,
            batch_size=self.opts.batch_size,
            shuffle=self.opts.shuffle_data,
            num_workers=num_workers,
            # pin_memory=opts.pin_memory,
        )
        return train_dataloader
    
    def val_dataloader(self):
        num_workers = multiprocessing.cpu_count()
        val_dataset_loader = DataLoader(
            dataset=self.dataset_dev ,
            batch_size=self.opts.batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        return val_dataset_loader
    
    def test_dataloader(self):
        num_workers = multiprocessing.cpu_count()
        te_dataset_loader = DataLoader(
            dataset=self.test_dataset ,
            batch_size=self.opts.batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        return te_dataset_loader
    
    def predict_dataloader(self):
        num_workers = multiprocessing.cpu_count()
        tr_dataset_loader = DataLoader(
            dataset=self.dataset_tr ,
            batch_size=self.opts.batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        dev_dataset_loader = DataLoader(
            dataset=self.dataset_dev ,
            batch_size=self.opts.batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        te_dataset_loader = DataLoader(
            dataset=self.test_dataset ,
            batch_size=self.opts.batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        return [tr_dataset_loader, dev_dataset_loader, te_dataset_loader]


def get_all(path, ext="pkl"):
    file_names = glob.glob("{}/*.{}".format(path, ext))
    return file_names

def chunkWithMyFold(seq, opts):
    seq2 = [x.split("/")[-1].split(".")[0] for x in seq]
    seq2 = list(zip(seq,seq2))
    out = []
    fold_files = glob.glob(os.path.join(opts.fold_paths, "*fold"))
    for fold_file in fold_files:
        fold_list = []
        f_fold = open(fold_file, "r")
        lines_fold = f_fold.readlines()
        f_fold.close()
        lines_fold = [l.strip().split(".")[0] for l in lines_fold]
        for s_path, s in seq2:
            if s in lines_fold:
                fold_list.append(s_path)
                # print(s_path)
        out.append(fold_list)
        # print(lines_fold[:10])
        # print(seq2[:10])
    # First test, rest train
    aux = []
    for i in range(1, len(out)):
        aux.extend(out[i])
    res = [out[0], aux]

    return res

def check_splits(splits):
    for i in range(len(splits)):
        s = splits[i]
        for j in range(len(splits)):
            if i==j: continue
            s_j = splits[j]
            for file_s in s:
                if file_s in s_j:
                    print("File {} repeated. In split {} and {}".format(file_s, i, j))

def check_splits_2(tr, te):
    s = 0
    for t in tr:
        if t in te:
            s += 1
            # print("file {} from train is in test partition ".format(t))
    return s

def get_trainTest(opts):
    # first fold -> test, second fold -> train
    train, test = [], []
    # train
    f = open(opts.train_lst, "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.strip()
        if "." in line:
            line = line.split(".")[0]
        train.append(os.path.join(opts.data_path, f'{line}.pkl' ))
    # test
    f = open(opts.test_lst, "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.strip()
        if "." in line:
            line = line.split(".")[0]
        test.append(os.path.join(opts.data_path, f'{line}.pkl' ))
    # val
    if opts.do_val:
        val = []
        f = open(opts.val_lst, "r")
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.strip()
            if "." in line:
                line = line.split(".")[0]
            val.append(os.path.join(opts.data_path, f'{line}.pkl' ))
        return [test, train, val]
    else:
        return [test, train]

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def chunkWithFold(seq, opts):
    seq2 = [x.split("/")[-1].split(".")[0] for x in seq]
    seq2 = list(zip(seq,seq2))
    out = []
    fold_files = glob.glob(os.path.join(opts.fold_paths, "fold*txt"))
    for fold_file in fold_files:
        fold_list = []
        f_fold = open(fold_file, "r")
        lines_fold = f_fold.readlines()
        f_fold.close()
        lines_fold = [l.strip().split(".")[0] for l in lines_fold]
        for s_path, s in seq2:
            if s in lines_fold:
                fold_list.append(s_path)
                # print(s_path)
        out.append(fold_list)
        # print(lines_fold[:10])
        # print(seq2[:10])

    return out