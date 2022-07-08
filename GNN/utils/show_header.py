from __future__ import print_function
from builtins import range
import sys, pickle
import numpy as np
import cv2
import os,sys,inspect
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import networkx as nx
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
try:
    from data.conjugate import conjugate_nx
except:
    from conjugate import conjugate_nx

try:
    from data.page import TablePAGE
except:
    from page import TablePAGE

try:
    from utils.metrics import evalHungarian, jaccard_distance, computePRF
except:
    from metrics import evalHungarian, jaccard_distance, computePRF

aux1 = 156
aux2 = 177

dict_clases = {
            'CH': 0,
            'O': 1,
            'D': 1,
        }

color_classes = {
            'CH':  (70, 1, 155),
            'O': (0, 126, 254),
            'D': (0, 126, 254),
        }
dict_classes_inv = {v: k for k, v in dict_clases.items()}

def read_lines(fpath):
    f = open(fpath, "r")
    lines = f.readlines()
    f.close()
    lines = [x.strip() for x in lines]
    return lines

def get_all(path, ext="pkl"):
    file_names = glob.glob(os.path.join(path, "*.{}".format(ext)))
    return file_names

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_image(path):
    if os.path.exists(path+".jpg"):
        p = path+".jpg"
    elif os.path.exists(path+".JPG"):
        p = path+".JPG"
    elif os.path.exists(path+".png"):
        p = path+".png"
    elif os.path.exists(path+".PNG"):
        p = path+".PNG"

    image = cv2.imread(p)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = resize(image, size=size)
    # image = image.astype(np.float32)
    # image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

def resize(img, size=(1024,512)):
    return cv2.resize(img.astype('float32'), size).astype('int32')

def show_cells(drawing, title, dir_to_save, fname=""):
    """
    Show the image
    :return:
    """
    plt.title(title)
    plt.imshow(drawing)
    # plt.show()
    plt.savefig("{}/{}.jpg".format(dir_to_save, fname), format='jpg', dpi=800)
    # plt.savefig("{}/{}.jpg".format(dir_to_save, title))
    plt.close()



def read_results(fname):
    results = {}
    f = open(fname, "r")
    lines = f.readlines()
    f.close()
    for line in lines[1:]:
        id_line, label, prediction = line.split(" ")
        # fname, id_line = id_line.split("/")[-1].split("-")
        # print(id_line)
        label = int(label)
        hyp = np.exp(float(prediction.rstrip()))
        results[id_line] = (label, hyp )
    return results

def show_all_imgs(drawing, title="",path="", redim=None):
    """
    Show the image
    :return:
    """
    plt.title(title)
    fig, ax = plt.subplots(1, len(drawing))
    for i in range(len(drawing)):
        if redim is not None:
            a = resize(drawing[i], redim)
            # print(a.shape)
            ax[i].imshow(a)
        else:
            ax[i].imshow(drawing[i])
    plt.savefig("{}/{}.png".format(path, title), format='png', dpi=800)
    plt.show( dpi=900)
    plt.close()


def print_BBs(nodes, labels, cc,
              results_gt, file_name, dir_to_save, drawing, _fF=None, gt=False, errors=None, nodes_orig=[], edges_orig=[], labels_orig=[], type="col", edge_features_orig=[] ):
    radius = 10
    color = (255, 0, 0)
    shape = drawing.shape
    # drawing_hyp = np.copy(drawing)
    height, width = shape[0], shape[1]
    # file_name = file_name.
    count_fail = 0
    words_failed = []
    # print(len(nodes))
    # print(len(labels))
    # exit()
    THICKNESS = 4
    # nodes_deleted, groups = search_for(nodes, nodes_orig, edges_orig, labels_orig, type, edge_features_orig, labels)

    # for group in groups:
    #     for count, data_node in enumerate(nodes):
    #         label_node = labels[count][type]
    #         if label_node == group:
    #             for c in cc:
    #                 if label_node
    # print(cc)
    # print(ids)
    # print(nodes_deleted, len(nodes_deleted))

    colors_cc = get_color_cc(cc)
    size_text = 1
    tick_text = 2
    bb_per_group = {}
    for count, data_node in enumerate(nodes):
        label_node = labels[count][type]
        
        x, y, w, h, prob_node = data_node[:5]
        x = int(x*width)
        y = int(y*height)
        w = int(w*width)
        h = int(h*height)
        bb = [
            x-(w//2), y-(h//2),x+(w//2), y+(h//2)
        ]
        x_min, y_min, x_max, y_max = map(int, bb)
        c_idx = search_idx_in_cc(count, cc)
        # if c_idx == -1:
            # print("cc", cc)
            # print("problem with img {} - node {}".format(file_name, count))
            # exit()
            # continue
        
        l_group = bb_per_group.get(c_idx, [])
        l_group.append(bb)
        bb_per_group[c_idx] = l_group

        color_rect = colors_cc[c_idx]

        cv2.rectangle(drawing, (x_min, y_min), (x_max, y_max),
                      # color=(50, 128, 5),
                      # color=results.get(count, (0,0,0)),
                      color=color_rect,
                      thickness=3,
                      )
        cv2.putText(drawing, str(count), (x, y), cv2.FONT_HERSHEY_COMPLEX, size_text, (0, 0, 0), 1)
        # cv2.putText(drawing, str(c_idx), (x, y), cv2.FONT_HERSHEY_COMPLEX, size_text, (50, 0, 0), tick_text)
        # cv2.putText(drawing, " " + str(c_idx), (x+10, y+10), cv2.FONT_HERSHEY_COMPLEX, size_text, (50, 0, 0), tick_text)
    file_name = file_name.split("/")[-1]
    if _fF is not None:
        title = file_name + "_{}_cols {} F1".format(len(cc), _fF)
    else:
        title = file_name + "_{}_cols".format(len(cc))
    if gt:
        file_name = file_name + "_gt"
    
    if errors is not None:
        color_fp, color_fn = (255,0,0), (0,255,0)
        done = set()
        for i,j,type_error in errors:
            if (i,j) not in done:
                done.add((i,j))
            else:
                continue
            data_node_i, data_node_j = nodes[i], nodes[j]
            xi, yi = data_node_i[:2]
            xi = int(xi*width)
            yi = int(yi*height)
            xj, yj = data_node_j[:2]
            xj = int(xj*width)
            yj = int(yj*height)
            if type_error: # fp
                cv2.line(drawing, (xi, yi), (xj, yj), color_fp, THICKNESS)
            # else:
            #     cv2.line(drawing, (xi, yi), (xj, yj), color_fn, THICKNESS//2)
          

    for group, l_group in bb_per_group.items():
        xs = np.array([[x[0], x[2]] for x in l_group]).flatten()
        ys = np.array([[x[1], x[3]] for x in l_group]).flatten()
        color_rect = colors_cc[group]
        x_max, x_min = max(xs), min(xs)
        y_max, y_min = max(ys), min(ys)
        cv2.rectangle(drawing, (x_min, y_min), (x_max, y_max),
                      # color=(50, 128, 5),
                      # color=results.get(count, (0,0,0)),
                      color=color_rect,
                      thickness=3,
        )
        # break

    show_cells(drawing,  title=title, fname=file_name, dir_to_save=dir_to_save)
    return count_fail, words_failed

def get_dict_nodes_edges(nodes, edges):
    """
    for non directed graph
    :param edges:
    :return:
    """
    res = {}
    for i, _ in enumerate(nodes):
        res[i] = []
    for i,j,_ in edges:
        # i -> j
        aux = res.get(i, [])
        aux.append(j)
        res[i] = aux
        # j -> i
        aux = res.get(j, [])
        aux.append(i)
        res[j] = aux
    return res

def main():
    """
    Quick script to show mask images stored on pickle files
    """
    min_w = 0.5
    dir_to_save_local = "headers"
  
    data_path = sys.argv[1]
    files_to_use = "/data/HisClima/DatosHisclima/test.lst"
    if files_to_use is not None:
        files_to_use = read_lines(files_to_use)
    data_pkls = "/data/HisClima/DatosHisclima/graphs/graphs_preprocessed/graph_k10_wh0ww0jh10jw1_min0_maxwidth0.5"
  
    # NAF
    ##Cols
    # if type_ == "col":
    #     results = f"{data_path}/results_cols.txt"
    # elif type_ == "row":
    #     results = f"{data_path}/results_rows.txt"
    results = f"{data_path}/results.txt"


    dir_img = "/data/HisClima/DatosHisclima/data/GT-Corregido"
    # dir_img = "/data2/jose/corpus/tablas_DU/icdar19_abp_small/"
    file_list = get_all(data_pkls)
    # print(file_list)
    results = os.path.abspath(results)
    dir_to_save = "/".join(results.split("/")[:-1])
    dir_to_save = os.path.join(dir_to_save, dir_to_save_local)
    results = read_results(results)
    print("A total of {} lines classifieds in results".format(len(results)))

    gts, hyps = [],[]
    for i, [gt, hyp] in results.items():
        gts.append(gt)
        hyps.append(int(hyp>=min_w))
    acc = accuracy_score(gts, hyps)
    total = np.sum(np.array(gts)==np.array(hyps))
    print(f'A total of {total}/{len(gts)} well classigied - {acc} %')

    # TODO acabar
    create_dir(dir_to_save)
    print("Saving data on : {}".format(dir_to_save))
    # data_path = "/data/READ_ABP_TABLE/dataset111/graphs_preprocesseds/graphs_structure/graphs_structure_PI_fasttext_0.05_radius0.02"
    file_list = tqdm(file_list, desc="Files")

    file_search = "52709_004"
    file_names_ = [x.split("-edge")[0] for x in results.keys()]
    
    for raw_path in file_list:
        file_name_p = raw_path.split("/")[-1].split(".")[0]
        # raw_path_orig = os.path.join(data_path_orig, file_name_p+".pkl")
        # print(file_name_p, file_names_)
        if files_to_use is not None and file_name_p not in files_to_use:
            continue
        if file_name_p not in file_names_:
            continue
        # if file_search not in raw_path:
        #     continue

        # Read data from `raw_path`.
        # print("File: {}".format(raw_path))
        f = open(raw_path, "rb")
        data_load = pickle.load(f)
        f.close()
        ids = data_load['ids_tl']
        nodes = data_load['nodes']
        edges = data_load['edges']
        labels = data_load['labels']
       
        edge_features = data_load['edge_features']
        # print("A total of {} nodes and {} edges for {}".format(len(nodes), len(edges), raw_path))

     
        file_name = raw_path.split("/")[-1].split(".")[0]
        img = load_image(os.path.join(dir_img, file_name))

     
        
       

        gt_edges_graph_dict = None
        count_fail_, words_failed_ = print_BBs(nodes, labels, cc,
                                            gt_edges_graph_dict, file_name, dir_to_save, img, _fF, errors=errors, 
                                        #    nodes_orig=nodes_orig, edges_orig=edges_orig, labels_orig=labels_orig, type=type_, 
                                        #    edge_features_orig=edge_features_orig
                                            )
    # exit()
    fP, fR, fF = fP/len(file_list), fR/len(file_list), fF/len(file_list)
    # print("P: {} R: {} F1: {}".format( fP, fR, fF))
    fP, fR, fF = computePRF(nOk, nErr, nMiss)
    print("P: {} R: {} F1: {}".format( fP, fR, fF))
    print("Min group of cc: ", min_cc)
    # exit()


if __name__ == "__main__":
    main()
    # if len(sys.argv) > 1 and sys.argv[1] != "-h":
    #     main()
    # else:
    #     print("Usage: python {} <dir with GT pkl> <file with results (txt)> <dir with the REAL images to load>".format(sys.argv[0]))
