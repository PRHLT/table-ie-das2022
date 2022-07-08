from ast import expr_context
import os.path as osp
import os, re, time
import glob, pickle
import torch
from torch_geometric.data import Dataset, Data, DataLoader, InMemoryDataset
import numpy as np
try:
    from utils.optparse_graph import Arguments as arguments
except:
    import sys
    sys.path.append('../utils')
    from optparse_graph import Arguments as arguments
# from torch_geometric.utils import grid
# import spacy
import logging

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

class ABPDataset_Header(Dataset):
    """
    In order to create a torch_geometric.data.InMemoryDataset, you need to implement four fundamental methods:

    torch_geometric.data.InMemoryDataset.raw_file_names():
    A list of files in the raw_dir which needs to be found in order to skip the download.

    torch_geometric.data.InMemoryDataset.processed_file_names():
    A list of files in the processed_dir which needs to be found in order to skip the processing.

    torch_geometric.data.InMemoryDataset.download():
    Downloads raw data into raw_dir.

    torch_geometric.data.InMemoryDataset.process():
    Processes raw data and saves it into the processed_dir.
    """
    def __init__(self, root, split, flist, opts=None, transform=None, pre_transform=None):
        # super(ABPDataset_Header, self).__init__(root, transform, pre_transform, None)
        self.root = None
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = None
        self._indices = None
        self.len_ = 0

        self.opts = opts
        self.processed_dir_ = os.path.join(opts.work_dir, split)
        if not os.path.exists(self.processed_dir_):
            os.mkdir(self.processed_dir_)
        self.pre_transform = pre_transform
        self.flist = flist
        self.flist_processed = []
        # self.processed_dir = self.pathself.text_info = opts.text_info
        self.text_length = opts.text_length
        self.img_feats = opts.img_feats
        self.transform = transform
        self.min_nodes = 100
        if opts.conjugate.lower() == "span":
            self.min_nodes = 15

        # if opts.fasttext:
        #     self.model = ft.load_fasttext_format("/data2/jose/word_embedding/fastext/german/cc.de.300.bin")

        self.dict_clases = {
            'CH': 0,
            'O': 1,
            'D': 1,
        }
        self.num_classes = 2

        self.ids = []
        self.labels = []

        # POS Tagging etc from SPACY
        self.POS_tag = {
            'SPACE': 0, 'NUM': 1, 'PROPN': 2, 'NOUN': 3, 'X': 4, 'PART': 5, 'ADP': 6, 'CONJ': 7, 'PUNCT': 8, 'ADV': 9,
            'DET': 10, 'VERB': 11, 'SCONJ': 12, 'ADJ': 13, 'AUX': 14, 'PRON': 15,

        }
        self.ent_type = {
            '': 0, 'MISC': 1, 'ORG': 2, 'LOC': 3, 'PER': 4,
        }

        self.preprocessed = not self.opts.not_preprocessed
        
        # if self.text_info and not self.preprocessed:
        #     self.nlp = spacy.load('de')

    def get_prob_class(self):
        type_edge = self.opts.conjugate.lower()
        class_sum = np.zeros([2])
        if type_edge in ["row", "cell", "col"]:
            total = len(self.labels_edge)
            positives = np.sum(self.labels_edge)
            class_sum[0] = 1-(positives / total)
            class_sum[1] = positives / total
            return class_sum
        else:
            total = 0
            labels = []
            for fname in self.flist:
                with open(fname, "rb") as f:
                    sample = pickle.load(f)
                    if len(sample['nodes']) < self.min_nodes:
                        continue
                for label in sample['labels']:
                    # labels.append(self.dict_clases[label['DU_header']])
                    labels.append(self.dict_clases.get(label['DU_header'], label['DU_header']))
                _, counts = np.unique(labels,return_counts=True)
                if len(counts) == 2:
                    class_sum += counts
                else:
                    class_sum[0] += counts[0]
                total += len(labels)
            class_sum = class_sum / total
            return class_sum

    @property
    def raw_file_names(self):
        # return self.flist
        return []
    @property
    def processed_file_names(self):
        # return self.flist
        return []

    def len(self):
        """
        Returns the number of examples in your dataset.
        :return:
        """
        # print(self.processed_file_names)
        return self.len_

    def download(self):
        # Download to `self.raw_dir`.
        # self.raw_paths = []
        print("Download?")

    # def get_edge_info(self, edges):
    #     edge_index, info_edges = [], []
    #     for (i,j), info in edges:
    #         edge_index.append((i,j))
    #         info_edges.append(info)
    #     return np.array(edge_index).T, info_edges

    def get_positions(self, nodes):
        res = []
        for node in nodes:
            res.append(node[:2])
        return res

    def get_nodes(self, nodes):
        """Not used if its preprocessed"""

        if self.text_length and self.text_info and self.img_feats:
            return nodes

        res = []
        POS_len = len(list(self.POS_tag))
        ent_type_len = len(list(self.ent_type))
        text_len = POS_len + ent_type_len
        img_len = 300*50*3
        for node in nodes:

            len_n = len(node)
            hasta = len_n - (text_len + 1 + img_len)

            n = node[:hasta]
            text_feats = node[hasta:hasta+text_len]
            len_text = node[hasta+text_len:hasta+text_len+1]
            img = node[hasta+text_len+1:]

            if self.text_length:
                n.extend(len_text)
            if self.text_info:
                n.extend(text_feats)
            # if self.fasttext:
            #
            #     try:
            #         res_ft = self.model.wv[w][0]
            #     except:
            #         res_ft = np.zeros((300))
                # self.model =
            elif self.img_feats:
                n.extend(img)

            res.append(n)
        return res

    def create_groups_span(self, gts:dict):
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

    def generate_labels_edges(self, edges, labels, gts_span={}, connections_RO={}, opts=None, ids=[]):
        new_labels_cols = []
        new_labels_rows = []
        new_labels_cells = []
        new_labels_spans = []
        # print(gts_span)
        # print(connections_RO)
        ids_2 = [i.split(".xml-")[-1] for i in ids ]  #TODO verificar que funcione para todos los corpus]
        # ids_2 = [i.split("-")[-1] for i in ids ]  #TODO verificar que funcione para todos los corpus]
        # print(ids_2.index("line_1587565380629_13067"))
        # print(ids_2.index("line_1587565414738_13094"))
        # print(ids_2.index("line_1587565350476_13029"))
        # print(edges)
        for num_node, (i, j) in enumerate(edges):
            # Label
            row_i = labels[i]['row']
            col_i = labels[i]['col']

            row_j = labels[j]['row']
            col_j = labels[j]['col']
            # if i in list_idx or j in list_idx:
            #     print("Col from origin {} col from target {} ({} -> {})".format(col_i, col_j, i, j))

            if row_i == row_j and row_i != -1 and row_j != -1:
                new_labels_rows.append(1)
            else:
                new_labels_rows.append(0)

            if col_i == col_j and col_i != -1 and col_j != -1:
                new_labels_cols.append(1)
            else:
                new_labels_cols.append(0)

            same_cell = row_i == row_j and row_i != -1 and row_j != -1 and col_i == col_j and col_i != -1 and col_j != -1
            if same_cell:
                new_labels_cells.append(1)
            else:
                new_labels_cells.append(0)
            ##SPAN
            # gts_i = gts_span_dict.get((row_i, col_i))
            # gts_j = gts_span_dict.get((row_j, col_j))
            # if gts_i == gts_j:
            #     new_labels_spans.append(1)
            # else:
            #     new_labels_spans.append(0)
            if opts.directed_spans == "NO":
                gts_i = gts_span.get((row_i, col_i))
                if gts_i and ([row_j, col_j] in gts_i or same_cell) and (i,j):
                    new_labels_spans.append(1)
                else:
                    new_labels_spans.append(0)
            else:
                id_i = ids_2[i]
                connections = connections_RO.get(id_i, [])
                # print(id_i, connections)
                added = False
                # print(i, id_i)
                # if i == 6:
                #     print(connections)
                for id_c in connections:
                    try:
                        pos_c = ids_2.index(id_c)
                    except:
                        pos_c = -1
                    # print(" ?> ", j, id_c, pos_c)
                    # if j == 23:
                    #     print(i, j)
                    if pos_c == j:
                        new_labels_spans.append(1)
                        added = True
                        # print("--------------------------------------------------------------------------- ", id_i, id_c, i, j)
                        # exit()
                        break
                if not added:
                    new_labels_spans.append(0)
               
        # exit()
        #     print(i, j,  new_labels_spans[-1])
        # print(new_labels_spans)
        # exit()
        return new_labels_cells, new_labels_cols, new_labels_rows, new_labels_spans

    def get_neighbors_feats(self, nodes):
        res = []
        for node in nodes:
            n = node[:-1]
            res.append(n)
        return res

    def process(self):
        self._process()

    def _process(self):
        i = 0
        # if not os.path.exists(osp.join(self.root, 'processed')):
        #     os.mkdir(osp.join(self.root, 'processed'))
        self.labels_edge = []
        for raw_path in self.flist:
            # Read data from `raw_path`.
            # if "vol003_003" not in raw_path:
            #     continue
            fname = raw_path.split("/")[-1].split(".")[0]
            f = open(raw_path, "rb")
            data_load = pickle.load(f)
            f.close()
            ids = data_load['ids_tl']
            gts_span = data_load.get("gts_span", [])
            connections_RO = data_load.get("connections_RO", {})
            if not connections_RO:
                connections_RO = {}
            labels = []
            for label in data_load['labels']:
                try:
                    l = self.dict_clases.get(label['DU_header'], label['DU_header'])
                except:
                    l = label
                labels.append(l)
            # print(np.unique(labels, return_counts=True))
            # edge_index = torch.tensor(np.array(data_load['edges']).T, dtype=torch.long)
            # edge_index, info_edges = self.get_edge_info(data_load['edges'])
            edge_index = np.array(data_load['edges']).T
            info_edges = data_load['edge_features']
            # print(len(info_edges), len(info_edges[0]))
            # info_edges = np.zeros_like(data_load['edge_features'])
            nodes = data_load['nodes']
            nodes = np.array(nodes)
            info_edges = np.array(info_edges)
            # print(nodes.shape)
            # print(info_edges.shape)
            # exit()
            # print("raw_path {} nodes {}".format(raw_path, len(nodes)))
            if not self.preprocessed:
                nodes = self.get_nodes(nodes)
            # print(raw_path, nodes.shape)
            if len(nodes) < self.min_nodes:
                print(f'{fname} skipped')
                continue
            # info_edges = self.get_neighbors_feats(info_edges)
            positions = self.get_positions(nodes)
            self.labels.extend(labels)
            if len(nodes) != len(labels):
                print("Problem with nodes and labels")
                exit()
            if len(edge_index) > 0 and edge_index.shape[1] != len(info_edges):
                print(edge_index.shape[1])
                print(edge_index[:5])
                print(len(info_edges))
                print(info_edges[:5])
                print("Problem with edges")
                exit()
            # nodes_aux = nodes
            # nodes = []
            # for node in nodes_aux:
            #     nodes.append(node[-300:])
                # print(len(nodes[0]))
                # exit()
            # nodes = np.array(nodes)[:,1:2] # ONLY Y
            # nodes = np.ones_like(nodes)
            # info_edges = np.ones_like(info_edges) # NO EDGE FEATURES!
            # print(nodes.shape)
            # exit()
            type_edge = self.opts.conjugate.lower()
            # print(type_edge)
            if type_edge in ["row", "cell", "col", "span"]:
                labels_cells, labels_cols, labels_rows, labels_spans = self.generate_labels_edges(edge_index.T, data_load['labels'], gts_span, connections_RO, self.opts, ids)
                if type_edge == "row":
                    labels = labels_rows
                elif type_edge == "col":
                    labels = labels_cols
                elif type_edge == "cell":
                    labels = labels_cells
                elif type_edge == "span":
                    labels = labels_spans
                # self.ids.extend([fname]*len(labels))
                for (s,d) in edge_index.T:
                    s_id = ids[s]
                    d_id = ids[d]
                    if "xml-" in s_id:
                        s_id = s_id.split("xml-")[-1]
                        d_id = d_id.split("xml-")[-1]
                    id_line = "{} {} {}".format(fname, s_id,d_id)
                    self.ids.append(id_line)
                
                self.labels_edge.extend(labels)
            else:
                
                ids = [f"{fname} {x.split('xml-')[-1]}" for x in ids]
                # print(ids)
                # id_line = "{} {} {}".format(fname, s_id,d_id)
                self.ids.extend(ids)
            if "vol003_003" in raw_path:
                print("************", ids)
            

            #### test features
            

            ####

            x = torch.tensor(nodes, dtype=torch.float)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            # labels_edge = torch.tensor(labels_edge, dtype=torch.long)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            info_edges = torch.tensor(info_edges, dtype=torch.float)
            positions = torch.tensor(positions, dtype=torch.float)
            # exit()
            data = Data(x=x,
                        edge_index=edge_index,
                        y=labels_tensor,
                        edge_attr=info_edges,
                        pos=positions,
                        )
            self.len_ += 1


            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir_, 'data_{}.pt'.format(i)))
            self.flist_processed.append(os.path.join(self.processed_dir_, 'data_{}.pt'.format(i)))
            i += 1
        if self.opts.fix_class_imbalance:
            self.class_weights = self.get_prob_class()
            self.prob_class = self.class_weights
        # exit()
    def get(self, idx):
        """Implements the logic to load a single graph.
        Internally, torch_geometric.data.Dataset.__getitem__() gets data objects
        from torch_geometric.data.Dataset.get() and optionally transforms them according to transform.
        """
        data_save = torch.load(osp.join(self.processed_dir_, 'data_{}.pt'.format(idx)))
        if self.transform:
            data_save = self.transform(data_save)
        return data_save


def get_all(path, ext="pkl"):
    if path[-1] != "/":
        path = path+"/"
    file_names = glob.glob("{}*.{}".format(path,ext))
    return file_names

if __name__ == "__main__":
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


        fh = logging.FileHandler(opts.log_file, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # --- restore ch logger to INFO
        ch.setLevel(logging.INFO)

        return logger, opts

    logger, opts = prepare()
    opts.tr_data = "/data/READ_ABP_TABLE/dataset111/graphs_preprocesseds/graphs_structure_PI_lenText_0.05/"
    opts.fix_class_imbalance = True
    dataset_tr = ABPDataset_Header(root=opts.tr_data, split="dev", flist=get_all(opts.tr_data), transform=None, opts=opts)
    logger.info(dataset_tr.class_weights)
