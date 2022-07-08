#!/usr/bin/env python3.6
from __future__ import print_function
from __future__ import division
import time, os, logging, errno

from rdflib import Graph
from utils.functions import check_inputs_graph, save_checkpoint
from utils.optparse_graph import Arguments as arguments
from utils import metrics
import numpy as np
import torch, random, glob
from torch_geometric.data import DataLoader
from torch_geometric import transforms as T
import torch.nn.functional as F
from data import transforms_Graph
from models import model_pl
# from models import models_p2pala as models_p2pala_2AA, operations
import matplotlib.pyplot as plt
import cv2
from utils import functions
import shutil
from sklearn.model_selection import train_test_split
from utils.metrics import evaluate_graph_IoU
from torchcontrib.optim import SWA
import functools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import data.transforms as transforms
import torch
import multiprocessing
import os
import random
import numpy as np
# from models.ctc_loss import CTCLoss
# from models.ctc_greedy_decoder import CTCGreedyDecoder
import torch
import tqdm
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from data import GraphDataset
from torch import nn, save
from utils.functions import save_to_file, load_last_checkpoint_pl
import time
from utils.functions import create_dir, save_results, save_results_header
from main_graph import tensor_to_numpy

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def prepare():
    """
    Logging and arguments
    :return:
    """

    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    # --- keep this logger at DEBUG level, until aguments are processed
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # --- Get Input Arguments
    in_args = arguments(logger)
    opts = in_args.parse()
    if check_inputs_graph(opts, logger):
        logger.critical("Execution aborted due input errors...")
        exit(1)

    fh = logging.FileHandler(opts.log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # --- restore ch logger to INFO
    ch.setLevel(logging.INFO)
    return logger, opts

def process_predictions(dt):
    edge_index_total, outputs, gts = [], [], []
    for x in dt:
        o = x['outputs']
        outputs.extend(tensor_to_numpy(o[:,1]))
        gts.extend(tensor_to_numpy(x['y_gt']))
        edges = tensor_to_numpy(x["edge_index"]).T
        edge_index_total.extend(edges)
    outputs = np.array(outputs)
    gts = np.array(gts)
    return edge_index_total, outputs, gts

def main():
    global_start = time.time()
    logger, opts = prepare()
    # --- set device
    device = torch.device("cuda:{}".format(opts.gpu) if opts.use_gpu else "cpu")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # --- Init torch random
    # --- This two are suposed to be merged in the future, for now keep boot
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    input_channels = opts.input_channels
    do_train = True
    k_steps=1
    exp_name=opts.exp_name
    checkpoint_load = load_last_checkpoint_pl(opts.work_dir)
    logger.info(checkpoint_load)
    logger_csv = CSVLogger(opts.work_dir, name=exp_name)
    wandb_logger = WandbLogger(project=exp_name)
    path_save = os.path.join(opts.work_dir, "checkpoints")


    graphDataset = GraphDataset(opts=opts, logger=logger)

    checkpoint_callback = ModelCheckpoint(dirpath=opts.work_dir, save_top_k=1, 
                            monitor="val_loss",
                            # filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
                            auto_insert_metric_name=True
                        )
    net = model_pl.Net(   opts, graphDataset, logger)

    if checkpoint_load:
        net = net.load_from_checkpoint(checkpoint_load,  opts=opts, graphDataset=graphDataset, logger=logger)
        logger.info("Model loaded")
    net.to(device)
    wandb_logger.watch(net)

    trainer = pl.Trainer(min_epochs=opts.epochs, max_epochs=opts.epochs, logger=[logger_csv, wandb_logger], #wandb_logger
                    callbacks=[checkpoint_callback],
                    default_root_dir=path_save,
                    gpus=opts.gpu+1,
                    log_every_n_steps=k_steps,
                )
    if do_train:
        trainer.fit(net, graphDataset, ckpt_path=checkpoint_load,)

    
    ##TEST
    logger.info("TEST")
    IoU = True
    results_test = trainer.test(net, graphDataset, ckpt_path="best")
    outputs = trainer.predict(net, graphDataset, ckpt_path="best") # , ckpt_path='best'
    tr,val,te = outputs
    
    if IoU and opts.classify != "HEADER":
        edge_index_total, outputs, gts = process_predictions(te)
        results_tests = list(zip(graphDataset.test_dataset.ids, gts, outputs, edge_index_total))
        edge_index_total, outputs, gts = process_predictions(tr)
        results_train = list(zip(graphDataset.dataset_tr.ids, gts, outputs, edge_index_total))
        edge_index_total, outputs, gts = process_predictions(val)
        results_val = list(zip(graphDataset.dataset_dev.ids, gts, outputs, edge_index_total))

        """TEST"""
        fP_1, fR_1, fF_1, res_1 = metrics.evaluate_graph_IoU(graphDataset.list_te, results_tests, min_w=opts.min_prob, th=1.0, type_=opts.conjugate, pruned=opts.do_prune, conjugate=False, ORACLE=False)
        save_results(results_tests,results_train, results_val, res_1, opts.work_dir, logger)
    else:
        edge_index_total, outputs, gts = process_predictions(te)
        results_tests = list(zip(graphDataset.test_dataset.ids, gts, outputs))
        edge_index_total, outputs, gts = process_predictions(tr)
        results_train = list(zip(graphDataset.dataset_tr.ids, gts, outputs))
        edge_index_total, outputs, gts = process_predictions(val)
        results_val = list(zip(graphDataset.dataset_dev.ids, gts, outputs))
        save_results_header(results_tests, results_train, results_val, opts.work_dir, logger)

if __name__ == "__main__":
    main()