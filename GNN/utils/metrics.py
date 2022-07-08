import numpy as np
import random
import sklearn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from numpy import zeros, inf, array, argmin
import os
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import functools, time
# os.sysconf()
import networkx as nx, pickle, glob
try:
    from data.conjugate import conjugate_nx
except:
    from conjugate import conjugate_nx
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.seterr(divide='ignore', invalid='ignore')

def eval_accuracy(gt, hyp):
    """
    Eval accuracy on batch
    :param gt:
    :param hyp:
    :return:
    """
    hyp = np.where(hyp > 0.5, 1, 0)
    gt = np.where(gt > 0.5, 1, 0)
    acc = []
    p = []
    r = []
    f1 = []
    # accuracy: (tp + tn) / (p + n)
    for i in range(len(gt)):

        accuracy = accuracy_score(hyp[i], gt[i])

        # precision tp / (tp + fp)
        precision = precision_score(hyp[i], gt[i], zero_division=0)
        # recall: tp / (tp + fn)
        recall = recall_score(hyp[i], gt[i], zero_division=0)
        # f1: 2 tp / (2 tp + fp + fn)
        f1_ = f1_score(hyp[i], gt[i], zero_division=0)
        acc.append(accuracy)
        p.append(precision)
        r.append(recall)
        f1.append(f1_)

    return acc, p, r, f1

def timing(func):
    """timefunc's doc"""

    @functools.wraps(func)
    def time_closure(*args, **kwargs):
        """time_wrapper's doc string"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start
        print(f"Function: {func.__name__}, Time: {time_elapsed}")
        return result

    return time_closure

# @timing
def eval_graph(gt, hyp):
    """
    Eval accuracy on batch
    :param gt:
    :param hyp:
    :return:
    """
    hyp = np.exp(hyp)
    hyp = np.where(hyp > 0.5, 1, 0)
    # gt = np.where(gt > 0.5, 1, 0)
    # accuracy: (tp + tn) / (p + n)
    # print(gt)
    # print(hyp)
    # accuracy = accuracy_score(hyp, gt)

    # precision tp / (tp + fp)
    precision = precision_score(gt, hyp, average='macro', zero_division=0)
    # recall: tp / (tp + fn)
    recall = recall_score(gt, hyp, average='macro', zero_division=0)
    # f1: 2 tp / (2 tp + fp + fn)
    f1_ = f1_score(gt, hyp, average='macro', zero_division=0)
    TN, FP, FN, TP = confusion_matrix(gt, hyp).ravel()
    # print(TN, FP, FN, TP)
    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    # (recall_posivite*number_positve)+(recall_negative*number_negative)/(number_positive + number_negativ) = 0.5*50+0.5*50/(50+50) = 50/100 = 0.5

    # wACC = ((TP*(TP+FN))+(TN+(TN+FP)))/(TP+FP+FN+TN)
    auc_score = sklearn.metrics.roc_auc_score(gt, hyp)
    # sample_weight = []
    # for g in gt:
    #     if g == 1:
    #         sample_weight.append(0.1)
    #     else:
    #         sample_weight.append(1)
    # acc_balanced = sklearn.metrics.accuracy_score(gt, hyp, sample_weight=sample_weight)
    # fps = 0
    # for i, g in enumerate(gt):
    #     hyp_i = hyp[i]
    #     if g == 0 and hyp_i == 1:
    #         fps+=1
    # print(f'10*FPR+FNR+FDR {FPR} {FNR} {FDR} {(10*FPR)+FNR+FDR}, -> fps {fps} / {len(gt)}')

    # print(recall, precision, f1_)
    return ACC, precision, recall, f1_, FPR, TNR, TPR, PPV, NPV, FNR, FDR, auc_score


def computePRF(nOk, nErr, nMiss):
    eps = 0.00001
    fP = nOk / (nOk + nErr + eps)
    fR = nOk / (nOk + nMiss + eps)
    fF = 2 * fP * fR / (fP + fR + eps)
    return fP, fR, fF

def jaccard_distance(x,y):
    """
        intersection over union
        x and y are of list or set or mixed of
        returns a cost (1-similarity) in [0, 1]
    """
    try:    
        setx = set(x)
        return  1 - (len(setx.intersection(y)) / len(setx.union(y)))
    except ZeroDivisionError:
        return 0.0

def evalHungarian(X,Y,thresh, func_eval=jaccard_distance):
        """
        https://en.wikipedia.org/wiki/Hungarian_algorithm
        """          
        cost = [func_eval(x,y) for x in X for y in Y]
        cost_matrix = np.array(cost, dtype=float).reshape((len(X), len(Y)))
        r1,r2 = linear_sum_assignment(cost_matrix)
        toDel=[]
        for a,i in enumerate(r2):
            # print (r1[a],ri)      
            if 1 - cost_matrix[r1[a],i] < thresh :
                toDel.append(a)                    
        r2 = np.delete(r2,toDel)
        r1 = np.delete(r1,toDel)
        _nOk, _nErr, _nMiss = len(r1), len(X)-len(r1), len(Y)-len(r1)
        return _nOk, _nErr, _nMiss

def get_all(path, ext="pkl"):
    file_names = glob.glob("{}*.{}".format(path,ext))
    return file_names

def read_results(fname, conjugate=True):
    """
    Since the differents methods tried save results in different formats,
    we try to load all possible formats.
    """
    results = {}
    if conjugate:
        if type(fname) == str: 
            f = open(fname, "r")
            lines = f.readlines()
            f.close()
            for line in lines[1:]:
                id_line, label, prediction = line.split(" ")
                id_line = id_line.split("/")[-1].split(".")[0]
                results[id_line] = (int(label), np.exp(float(prediction.rstrip())) )
        else:
            for id_line, label, prediction in fname:
                id_line = id_line.split("/")[-1].split(".")[0]
                results[id_line] = int(label), np.exp(float(prediction))
    else:
        for fname, label, prediction, (i,j) in fname:
            id_line = fname
            results[id_line] = int(label), np.exp(float(prediction))
    return results

def create_groups_span(gts:dict):
    ngroup = 0
    res = {}
    for key, list_v in gts.items():
        row, col = key
        # list_v.append(key)
        group_k = res.get((row, col), None)
        if group_k is None:
            group_k = ngroup
            ngroup += 1
            res[(row, col)] = group_k
        for k in list_v:
            row, col = k
            res[(row, col)] = group_k
    return res

def evaluate_graph_IoU(file_list, results, min_w = 0.5, th = 0.8, type_="COL", conjugate=True, all_edges=False, pruned=False, ORACLE = False):
    
    type_ = type_.lower()
    if type(results) != dict:
        results = read_results(results, conjugate=conjugate and not all_edges)
    nOk, nErr, nMiss = 0,0,0
    fP, fR, fF = 0,0,0
    res = []
    blank_imgs = 0
    for raw_path in file_list:  
        # if fname_search not in raw_path:
        #     continue
        f = open(raw_path, "rb")
        data_load = pickle.load(f)
        f.close()
        ids = data_load['ids']
        ids_tl = data_load['ids_tl']
        if ids_tl and "xml-" in ids_tl[0]:
            ids_tl = [idd.split("xml-")[-1] for idd in ids_tl]
        nodes = data_load['nodes']
        edges = data_load['edges']
        labels = data_load['labels']
        edge_features = data_load['edge_features']   
        gts_span = data_load["gts_span"]
        min_nodes = 100
        if type_ == "span":
            gts_span_dict = create_groups_span(gts_span)
            min_nodes = 15
        if len(nodes) < min_nodes:
            continue
        if conjugate:
            if type_ == "span":
                ids, new_nodes, new_edges, new_labels, new_edge_feats, ant_nodes, ant_edges = data_load["conjugate"]
                # for k,v in gts_span.items():
                #     print(k, v)
                # print("----")
                # print(gts_span_dict)
                # exit()
            elif data_load.get("conjugate", None) is not None:
                    ids, new_nodes, new_edges, new_labels_cells, new_labels_cols, new_labels_rows, new_edge_feats, ant_nodes, ant_edges = data_load["conjugate"]
                    # print(nodes)
            else:
                ids, new_nodes, new_edges, new_labels_cells, new_labels_cols, new_labels_rows, \
                new_edge_feats, ant_nodes, ant_edges = conjugate_nx(ids,
                        nodes, edges, labels, edge_features, idx=None, list_idx=None)
            if len(edges) != len(new_nodes):
                print("Problem with {} {} edges and {} new_nodes".format(raw_path, len(edges), len(new_nodes)))
                #continue
        else:
            ant_nodes = edges
        file_name = raw_path.split("/")[-1].split(".")[0]

        G = nx.Graph()
        G_gt = nx.Graph()
        weighted_edges_GT = []
        weighted_edges = []
        weighted_edges_dict_gt = {}
        out_graph = []
        for i, node in enumerate(nodes):
            # G.add_node(i, attr=node)
            if type_ == "cell":
                r, c = labels[i]['row'], labels[i]['col']
                if r == -1 or c == -1:
                    ctype = -1
                else:
                    ctype = "{}_{}".format(r,c)
            elif type_ == "span":
                # ctype = labels[i][type_]
                r, c = labels[i]['row'], labels[i]['col']
                if r == -1 or c == -1:
                    ctype = -1
                else:
                    ctype = gts_span_dict.get((r,c))
            else:
                ctype = labels[i][type_]
            # id_ = ids[i]

            if ctype != -1:
                G.add_node(i)
                G_gt.add_node(i)
                cc_type = weighted_edges_dict_gt.get(ctype, set())
                cc_type.add(i)
                weighted_edges_dict_gt[ctype] = cc_type
            else:
                out_graph.append(i)


        aux_dict = {}
        if all_edges:

            for i, node_i in enumerate(nodes):
                for j, node_j in enumerate(nodes):

                    x_coord_o, y_coord_o = node_i[0], node_i[1]
                    x_coord_t, y_coord_t = node_j[0], node_j[1]
                    dif_y = abs(y_coord_o - y_coord_t)
                    dif_x = abs(x_coord_o - x_coord_t)
                    key = "{}_edge{}-{}".format(file_name, i, j)
                    try:
                        gt, hyp_prob = results.get(key)
                    except Exception as e:
                        print("Problem with key {} (no conjugation - all_edges)".format(key))
                        # continue
                        raise e
                    if ORACLE:
                        if i in out_graph or j in out_graph: continue
                        if gt:
                            hyp_prob = 1
                            weighted_edges.append((i,j,hyp_prob))
                        # else:
                        #     hyp_prob = 0
                    else:
                        # if hyp_prob > min_w and dif_x <= 0.01:
                        if hyp_prob > min_w:
                            weighted_edges.append((i,j,hyp_prob))
        elif not all_edges and not conjugate:
            for (i,j) in edges:
                id_i = ids_tl[i]
                id_j = ids_tl[j]
                key = "{} {} {}".format(file_name, id_i, id_j)
                key2 = "{} {} {}".format(file_name, id_j, id_i)
                try:
                    gt, hyp_prob = results.get(key, results.get(key2))
                    # print(gt, hyp_prob)
                except Exception as e:
                    # print(list(results.keys())[:10])
                    # for x in list(results.keys()):
                    #     if "vol003_003" in x:
                    #         print(x)
                    print("Problem with key {} - {}".format(key, key2))
                    # continue
                    raise e
                if ORACLE:
                        if i in out_graph or j in out_graph: continue
                        if gt:
                            hyp_prob = 1
                            weighted_edges.append((i,j,hyp_prob))
                        # else:
                        #     hyp_prob = 0
                else:
                    # if hyp_prob > min_w and dif_x <= 0.01:
                    if hyp_prob > min_w:
                        weighted_edges.append((i,j,hyp_prob))
        else:
            for idx, (i, j) in enumerate(ant_nodes):
                aux_dict[(i, j)] = idx
            count_acc = []
            for count, (i, j) in enumerate(edges):
                
                # idx_edge = aux_dict.get((i, j), aux_dict.get((j, i)))
                # if conjugate and type_ == "span":
                #     #TODO
                #     pass
                # el
                if conjugate:
                    idx_edge = "{}_{}".format(j, i)
                    key = "{}-edge{}".format(file_name, idx_edge)
                    idx_edge2 = "{}_{}".format(i, j)
                    key2 = "{}-edge{}".format(file_name, idx_edge2)
                    if pruned:
                        found = True
                        try:
                            gt, hyp_prob = results.get(key, results.get(key2))
                            # print((j,i), key, key2, "si")
                        except:
                            gt, hyp_prob = 0, 0 #TODO
                            found = False
                            # print((j,i), key, key2, "no")
                        # if gt:
                        #     hyp_prob = 1
                    else:
                        try:
                            gt, hyp_prob = results.get(key, results.get(key2))
                        except Exception as e:
                            print(key, key2)
                            raise e
                else:
                    # print(list(results.keys())[:10], file_name)
                    # exit()
                    key = "{}_edge{}-{}".format(file_name, i, j)
                    key2 = "{}_edge{}-{}".format(file_name, j, i)
                    try:
                        gt, hyp_prob = results.get(key, results.get(key2))
                    except Exception as e:
                        print("Problem with key {} - {}".format(key, key2))
                        # continue
                        raise e
                if ORACLE:
                   
                    if i in out_graph or j in out_graph: continue
                    if gt:
                        hyp_prob = 1
                        weighted_edges.append((i,j,hyp_prob))
                    # else:
                    #     hyp_prob = 0
                else:
                    if hyp_prob > min_w:
                        weighted_edges.append((i,j,hyp_prob))
                # if gt:
                #     weighted_edges_GT.append((i,j,1))
                count_acc.append((hyp_prob > min_w) == gt)

        
        G.add_weighted_edges_from(weighted_edges)
        # cc = nx.connected_component_subgraphs(G)
        
        res_alg = nx.algorithms.community.label_propagation.asyn_lpa_communities(G)
        res_alg = [list(c) for c in res_alg]

        cc = (G.subgraph(c) for c in nx.connected_components(G))
        cc = [sorted(list(c)) for c in cc]

        cc_gt = [ sorted(list(ccs)) for ctype,ccs in weighted_edges_dict_gt.items()]

        cc = res_alg

        cc_gt.sort()
        cc.sort()

        _nOk, _nErr, _nMiss = evalHungarian(cc, cc_gt, th, jaccard_distance)
        _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)
        # print(_nOk, _nErr, _nMiss)
        # print(cc)
        # print(cc_gt)
        # exit()
        # if _fP < 0.99:
        #     print(file_name)
        res.append([raw_path,  _nOk, _nErr, _nMiss, _fP, _fR, _fF])
        if not cc_gt :
            blank_imgs += 1
            continue
        
        nOk += _nOk
        nErr += _nErr
        nMiss += _nMiss
        fP += _fP
        fR += _fR
        fF += _fF
        gt_edges_graph_dict = None
    len_files = len(file_list) - blank_imgs
    fP, fR, fF = fP/len_files, fR/len_files, fF/len_files
    # print("_nOk {}, _nErr {}, _nMiss {}, P: {} R: {} F1: {}".format(nOk, nErr, nMiss, fP, fR, fF))
    return fP, fR, fF, res

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

# @timing
def eval(opts, net, device, dataset, dataloader, list_files, conjugate, ORACLE=False, IoU=True):
    res_hyp, res_gt = [],[]
    edge_index_total = []
    
    for v_batch, v_sample in enumerate(dataloader):
        hyp = net(v_sample.to(device))
        if opts.g_loss == "NLL":
            hyp = hyp[:, 1]
        else:
            hyp = F.logsigmoid(hyp)
        hyp = tensor_to_numpy(hyp)
        y_gt = tensor_to_numpy(v_sample.y)
        if opts.classify == "EDGES":
            edges = tensor_to_numpy(v_sample.edge_index).T
            edge_index_total.extend(edges)
        res_hyp.append(hyp)
        res_gt.append(y_gt)
        del v_batch
        del v_sample
        del hyp
        del y_gt
    predictions = np.hstack(res_hyp)
    labels = np.hstack(res_gt)
    acc, p, r, f1, fpr, tnr, tpr, ppv, npv, fnr, fdr, auc_score = eval_graph(labels, predictions)
    if opts.classify == "EDGES":
        
        results_test = list(zip(dataset.ids, labels, predictions, edge_index_total))
    else:
        results_test = list(zip(dataset.ids, labels, predictions))
    # fP, fR, fF, res = evaluate_graph_IoU(list_train_val, results_test, min_w=opts.min_prob, th=0.8, type_=opts.conjugate, pruned=opts.do_prune, conjugate=conjugate)
    if IoU and opts.classify != "HEADER":
        fP_1, fR_1, fF_1, res_1 = evaluate_graph_IoU(list_files, results_test, min_w=opts.min_prob, th=1.0, type_=opts.conjugate, pruned=opts.do_prune, conjugate=conjugate, ORACLE=ORACLE)
        return  acc, p, r, f1, fpr, tnr, tpr, ppv, npv, fnr, fdr, auc_score, fP_1, fR_1, fF_1, res_1, results_test
    elif opts.classify == "HEADER":
        return acc, p, r, f1, fpr, tnr, tpr, ppv, npv, fnr, fdr, auc_score, None , None, None, None, results_test 
    else:
        return  acc, p, r, f1, fpr, tnr, tpr, ppv, npv, fnr, fdr, auc_score

if __name__ == "__main__":
    test_samples()
