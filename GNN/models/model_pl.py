import pytorch_lightning as pl
import torch
import numpy as np
import torchmetrics
from main_graph import tensor_to_numpy
from models.Graph import Net as GraphNet
from utils import metrics

class Net(pl.LightningModule):
    def __init__(self, opts, graphDataset, logger):
        super(Net, self).__init__()
        self.opts = opts
        self.logger_file=logger
        self.graphDataset = graphDataset
        self.dataset_tr = graphDataset.dataset_tr
        self.dataset_val = graphDataset.dataset_dev
        self.dataset_test = graphDataset.test_dataset
        # self.dataset_tr = dataset_tr
        # self.dataset_val = dataset_val
        # self.dataset_test = dataset_test
        self.net = GraphNet(self.dataset_tr, opts)

        # self.train_acc = torchmetrics.Accuracy()
        # self.val_acc = torchmetrics.Accuracy()
        # self.test_acc = torchmetrics.Accuracy()

        self.logger_file.info(f'Dataset num_node_feats {self.dataset_tr.num_node_features}, edges attr {self.dataset_tr.num_edge_features} last layer mlp edge classif {opts.layers[-1]}')
        ## Loss function
        c_weights = self.dataset_tr.prob_class
        self.logger_file.info("Using {} loss function".format(opts.g_loss))
        if opts.alpha_FP != 0:
            c_weights[0] = opts.alpha_FP * (1/(np.log(1+c_weights[0])))
            c_weights[1] = 1
        self.logger_file.info("Class weight : {}".format(c_weights))
        self.criterion = torch.nn.NLLLoss(reduction="mean", weight=torch.Tensor(c_weights))
    
    def forward(self, batch):
        # x_orig, edge_index_orig = data.x, data.edge_index
        # edge_attr_orig = data.edge_attr
        x = self.net(batch)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.opts.adam_lr, betas=(self.opts.adam_beta1, self.opts.adam_beta2)
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.opts.steps, gamma=self.opts.gamma_step)
        return [optimizer], [scheduler]
    
  
    def training_step(self, train_batch, batch_idx):
        # get the inputs
        hyp = self(train_batch)
        loss = self.criterion(hyp, train_batch.y)
     
        self.log('train_loss', loss)

        # self.train_acc(torch.exp(hyp), train_batch.y)
        # self.log('train_acc_step', self.train_acc)
      
        return {'loss': loss, 'outputs': hyp, 'y_gt':train_batch.y, "edge_index": train_batch.edge_index}

    def training_epoch_end(self, outs):
        # log epoch metric
        self.logger_file.info("\n   TRAIN\n")
        loss = np.mean([o['loss'].item() for o in outs])
        self.log('train_loss_epoch', loss)
        self.logger_file.info(f'Train loss {loss}')
        IoU = True
        conjugate = False
        ORACLE = False
        if self.current_epoch and self.current_epoch % self.opts.show_train == 0:
            
            outputs = []
            gts = []
            for x in outs:
                o = x['outputs']
                outputs.extend(tensor_to_numpy(o[:,1]))
                gts.extend(tensor_to_numpy(x['y_gt']))
            outputs = np.array(outputs)
            gts = np.array(gts)
            acc, p, r, f1, fpr, tnr, tpr, ppv, npv, fnr, fdr, auc_score = metrics.eval_graph(gts, outputs)
            self.logger_file.info(f'TRAIN EPOCH {self.current_epoch} - acc {acc} p {p} r {r} f1 {f1} fpr {fpr} auc_score {auc_score}')
            self.log('train_acc_epoch', acc)
            self.log('train_f1_epoch', f1)
            self.log('train_fpr_epoch', fpr)
            self.log('train_auc_score_epoch', auc_score)

            # if IoU and self.opts.classify != "HEADER":
            #     results_test = list(zip(self.dataset_tr.ids, labels, predictions, edge_index_total))
            #     fP_1, fR_1, fF_1, res_1 = metrics.evaluate_graph_IoU(list_files, results_test, min_w=self.opts.min_prob, th=1.0, type_=self.opts.conjugate, pruned=self.opts.do_prune, conjugate=conjugate, ORACLE=ORACLE)
            #     return  acc, p, r, f1, fpr, tnr, tpr, ppv, npv, fnr, fdr, auc_score, fP_1, fR_1, fF_1, res_1, results_test
    
    def validation_step(self, val_batch, batch_idx):

        hyp = self(val_batch)
        loss = self.criterion(hyp,val_batch.y)
        self.log('val_loss', loss)

        # self.val_acc(torch.exp(hyp), val_batch.y)
        # self.log('val_acc_step', self.val_acc)
      
        return {'loss': loss, 'outputs': hyp, 'y_gt':val_batch.y, "edge_index": val_batch.edge_index}
    
    def validation_epoch_end(self, outs):
        # log epoch metric
        self.logger_file.info("\n   VAL\n")

        loss = np.mean([o['loss'].item() for o in outs])
 
        self.log('val_loss_epoch', loss)
        self.logger_file.info(f'Val loss {loss}')
        IoU = True
        conjugate = False
        ORACLE = False
        if self.current_epoch and self.current_epoch % self.opts.show_val == 0:
            
            outputs = []
            gts = []
            edge_index_total = []
            for x in outs:
                o = x['outputs']
                outputs.extend(tensor_to_numpy(o[:,1]))
                gts.extend(tensor_to_numpy(x['y_gt']))
                edges = tensor_to_numpy(x["edge_index"]).T
                edge_index_total.extend(edges)
            outputs = np.array(outputs)
            gts = np.array(gts)
            acc, p, r, f1, fpr, tnr, tpr, ppv, npv, fnr, fdr, auc_score = metrics.eval_graph(gts, outputs)
            self.logger_file.info(f'VAL EPOCH {self.current_epoch} - acc {acc} p {p} r {r} f1 {f1} fpr {fpr} auc_score {auc_score}')
            self.log('val_acc_epoch', acc)
            self.log('val_f1_epoch', f1)
            self.log('val_fpr_epoch', fpr)
            self.log('val_auc_score_epoch', auc_score)

            if IoU and self.opts.classify != "HEADER":
                results_test = list(zip(self.dataset_val.ids, gts, outputs, edge_index_total))
                fP_1, fR_1, fF_1, res_1 = metrics.evaluate_graph_IoU(self.graphDataset.list_val, results_test, min_w=self.opts.min_prob, th=1.0, type_=self.opts.conjugate, pruned=self.opts.do_prune, conjugate=conjugate, ORACLE=ORACLE)
                # return  acc, p, r, f1, fpr, tnr, tpr, ppv, npv, fnr, fdr, auc_score, fP_1, fR_1, fF_1, res_1, results_test
                print(f'VAL EPOCH {self.current_epoch} - acc {acc} p {p} r {r} f1 {f1} - p@100 {fP_1} r@100 {fR_1} f1@100 {fF_1} fpr {fpr} auc_score {auc_score}')
                self.log('val_p@100', fP_1)
                self.log('val_r@100', fR_1)
                self.log('val_f1@100', fF_1)
    
    
    def test_step(self, test_batch, batch_idx):
        # get the inputs
        hyp = self(test_batch)
        
        loss = self.criterion(hyp,test_batch.y)
        self.log('test_loss', loss)

        # self.test_acc(hyp, test_batch.y)
        # self.log('test_acc_step', self.test_acc)

        return {'loss': loss, 'outputs': hyp, 'y_gt':test_batch.y, "edge_index": test_batch.edge_index}

    
    def test_epoch_end(self, outs):
        # log epoch metric
        self.logger_file.info("\n  TEST ")
        loss = np.mean([o['loss'].item() for o in outs])
        self.logger_file.info(f'Test loss {loss}')

        IoU = True
        conjugate = False
        ORACLE = False
            
        outputs = []
        gts = []
        edge_index_total = []
        for x in outs:
            o = x['outputs']
            outputs.extend(tensor_to_numpy(o[:,1]))
            gts.extend(tensor_to_numpy(x['y_gt']))
            edges = tensor_to_numpy(x["edge_index"]).T
            edge_index_total.extend(edges)
        outputs = np.array(outputs)
        gts = np.array(gts)
        acc, p, r, f1, fpr, tnr, tpr, ppv, npv, fnr, fdr, auc_score = metrics.eval_graph(gts, outputs)
        self.logger_file.info(f'TEST EPOCH {self.current_epoch} - acc {acc} p {p} r {r} f1 {f1} fpr {fpr} auc_score {auc_score}')
        self.log('test_acc_epoch', acc)
        self.log('test_f1_epoch', f1)
        self.log('test_fpr_epoch', fpr)
        self.log('test_auc_score_epoch', auc_score)

        if IoU and self.opts.classify != "HEADER":
            results_test = list(zip(self.dataset_test.ids, gts, outputs, edge_index_total))
            fP_1, fR_1, fF_1, res_1 = metrics.evaluate_graph_IoU(self.graphDataset.list_te, results_test, min_w=self.opts.min_prob, th=1.0, type_=self.opts.conjugate, pruned=self.opts.do_prune, conjugate=conjugate, ORACLE=ORACLE)
            # return  acc, p, r, f1, fpr, tnr, tpr, ppv, npv, fnr, fdr, auc_score, fP_1, fR_1, fF_1, res_1, results_test
            self.logger_file.info(f'TEST EPOCH {self.current_epoch} - acc {acc} p {p} r {r} f1 {f1} - p@100 {fP_1} r@100 {fR_1} f1@100 {fF_1} fpr {fpr} auc_score {auc_score}')
            self.log('test_p@100', fP_1)
            self.log('test_r@100', fR_1)
            self.log('test_f1@100', fF_1)
    
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # [tr_dataset_loader, dev_dataset_loader, te_dataset_loader]
        if dataloader_idx == 0:
            ids = self.dataset_tr.ids
        hyp = self(batch)
        ys = batch.y
        if batch.y is None:
            ys = torch.full_like(hyp, fill_value=-1)[:,1]
        # print("Predicting image ", filename, outputs.shape)
        return {'outputs': hyp, 'y_gt':ys, "edge_index": batch.edge_index, "dataloader_idx":dataloader_idx}