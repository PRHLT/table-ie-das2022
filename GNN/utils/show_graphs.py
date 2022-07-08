from __future__ import print_function, absolute_import
from builtins import range
import sys, pickle
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.append('..')
# from data.graph_dataset import ABPDataset_BIESO


try:
    from data.page import TablePAGE
except:
    from page import TablePAGE

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

def get_all(path, ext="pkl"):
    file_names = glob.glob(os.path.join(path, f'*.{ext}'))
    return file_names

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_image(path):
    print(path)
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

def show_cells(drawing, title, dir_to_save):
    """
    Show the image
    :return:
    """
    plt.title(title)
    plt.imshow(drawing)
    # plt.show()
    plt.savefig("{}/{}.jpg".format(dir_to_save, title), format='jpg', dpi=400)
    plt.close()



def read_results(fname):
    f = open(fname,     "r")
    lines = f.readlines()
    f.close()
    results = {}
    for line in lines[1:]:
        id_line, label, prediction = line.split(" ")
        results[id_line] = (int(label), int(prediction))
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
    plt.savefig("{}/{}.jpg".format(path, title), format='jpg', dpi=400)
    plt.show( dpi=900)
    plt.close()

def edge_to_dict(edges):
    res = {}
    for i,j in edges:
        aux = res.get(i, [])
        aux.append(j)
        res[i] = aux
    return res

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

def print_BBs(nodes, labels, edges, file_name, dir_to_save, drawing, conj='col', gts_span=None, show_labels=False):
    if gts_span is not None:
        gts_span_dict = create_groups_span(gts_span)
    radius = 50
    size_text = 2
    tick_text = 2
    color = (255, 0, 0)
    color_dest = (0, 100, 0)
    color_orig = (0, 0, 100)
    shape = drawing.shape
    height, width = shape[0], shape[1]
    # file_name = file_name.
    count_fail = 0
    words_failed = []
    edges_dict = edge_to_dict(edges)
    # print(edges)
    for count, data_node in enumerate(nodes):
        # if count not in [394, 39, 238]: continue
        if gts_span is not None:
            r, c = labels[count]['row'], labels[count]['col']
            col = gts_span_dict[(r,c)]
            print(f'  -> {count} | {r},{c}')
        else:
            col = int(labels[count][conj])
        x, y = data_node[:2]
        x = int(x*width)
        y = int(y*height)

        thickness = 6
        if col % 2 == 0:
            color = (255,0,0)
        else:
            color = (0,0,255)
        neighs = edges_dict.get(count, [])

        x2, y2 = data_node[2], data_node[3]
        x2 = int(x2 * width)
        y2 = int(y2 * height)

        # xs = np.array([[x[0], x[2]] for x in l_group]).flatten()
        # ys = np.array([[x[1], x[3]] for x in l_group]).flatten()
        # x_max, x_min = max(xs), min(xs)
        # y_max, y_min = max(ys), min(ys)
        color_rect =  (0, 255, 0)
        cv2.rectangle(drawing, (x, y), (x2, y2),
                    # color=(50, 128, 5),
                    # color=results.get(count, (0,0,0)),
                    color=color_rect,
                    thickness=3,
        )

        for neigh in neighs:
            # if count == 120:
            #     print(neigh)
            # elif neigh == 120:
            #     print(count)
            # else:
            #     continue
            links = nodes[neigh]
            x_d, y_d = links[:2]
            x_d = int(x_d * width)
            y_d = int(y_d * height)
            
            # if count == 104 or neigh == 104 :
            #     print(count, neigh, x_d, y_d)
                # print("--")

            # cv2.line(drawing, (x, y), (x_d, y_d), (0, 255, 0), thickness//2)

            # if 394 in neighs:
            #     print(data_node, neighs)
            # if count in [26,27]:
            if show_labels:
                cv2.putText(drawing, str(col), (x, y), cv2.FONT_HERSHEY_COMPLEX, size_text, (50, 0, 0), tick_text)
            else:
                cv2.putText(drawing, str(count), (x, y), cv2.FONT_HERSHEY_COMPLEX, size_text, (50, 0, 0), tick_text)

            # cv2.circle(drawing, (x, y), radius, color, thickness)
            
            # cv2.line(drawing, (x, y), (x_d, y_d), (0, 255, 0), thickness//2)
            



    show_cells(drawing,  title=file_name, dir_to_save=dir_to_save)
    return count_fail, words_failed


def main():
    """
    Quick script to show mask images stored on pickle files
    """
    data_path = sys.argv[1]
    dir_img = sys.argv[2]
    file_list = get_all(data_path)
    conj = 'col'
    show_labels = True
    dir_to_save = os.path.join(data_path, "results_{}".format(conj))
    if show_labels:
        dir_to_save += "_labels"
    create_dir(dir_to_save)
    # /data/HisClima/JeanneteAndAlbatrossHYP/graph_k10_wh4ww0jh1jw1_maxwidth0.5_minradio0.1
    # /data/HisClima/HisClimaProd/DLA/HisClima_0/images/
    # data_path = "/data/READ_ABP_TABLE/dataset111/graphs_preprocesseds/graphs_structure/graphs_structure_PI_lenText_0.05_radius0.02"
    # dataset_tr = ABPDataset_BIESO(root=data_path, split="dev", flist=file_list, transform=None, opts=None)
    file_list = tqdm(file_list, desc="Files")
    file_search = "_064_"
    for raw_path in file_list:
        if file_search not in raw_path:
            continue
        # Read data from `raw_path`.
        # print("File: {}".format(raw_path))
        f = open(raw_path, "rb")
        data_load = pickle.load(f)
        f.close()
        ids = data_load['ids']
        nodes = data_load['nodes']
        edges = data_load['edges']
        labels = data_load['labels']
        edge_features = data_load['edge_features']
        if conj == 'span':
            gts_span = data_load['gts_span']
        else:
            gts_span = None

        # ids, new_nodes, new_edges, new_labels_cells, new_labels_cols, new_labels_rows, new_edge_feats, ant_nodes, ant_edges = dataset_tr.conjugate_nx(ids,
        #         nodes, edges, labels, edge_features)
        file_name = raw_path.split("/")[-1].split(".")[0]
        img = load_image(os.path.join(dir_img, file_name))

        count_fail_, words_failed_ = print_BBs(nodes, labels, edges, file_name, dir_to_save, img, conj=conj, gts_span=gts_span, show_labels=show_labels)
        # exit()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "-h":
        main()
    else:
        print("Usage: python {} <dir with GT pkl> <dir with the REAL images to load>".format(sys.argv[0]))
