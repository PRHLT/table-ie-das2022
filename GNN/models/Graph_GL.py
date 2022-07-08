import torch.nn.functional as F
import torch_geometric.nn as geo_nn
import torch
from torch import nn
import torch_geometric as tg
# from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.utils import to_undirected
try:
    from models.nnconv import EdgeFeatsConv, EdgeFeatsConvMult
except:
    from nnconv import EdgeFeatsConv, EdgeFeatsConvMult


def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

class myNNConv(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(myNNConv, self).__init__()
        mlp = Seq(Linear(2 * in_c, out_c),
                  ReLU(),
                  Linear(out_c, out_c))
        self.k = opts.knn
        self.dynamic = self.k > 0
        self.NNConv = geo_nn.EdgeConv(mlp)
        self.bn = bn
        self.dropout = dropout
        if bn:
            self.batchNorm = nn.BatchNorm1d(out_c)
    def forward(self, data):
        x, edge_index, edge_attr = data
        if self.dynamic:
            print("dynamic, ", self.k)
            edge_index = geo_nn.knn_graph(x, self.k, 
            # batch=batch, 
            loop=False, cosine=False)
            edge_index = to_undirected(edge_index)
        if self.dropout:
            edge_index, edge_attr = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=0.1, force_undirected=True,
                                                     )
        x = self.NNConv(x, edge_index,)
        if self.bn:
            x = self.batchNorm(x)
        return x, edge_index, edge_attr

class EdgeFeatsConvNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(EdgeFeatsConvNN, self).__init__()
        root_weight = opts.root_weight
        num_edge_features = dataset.num_edge_features
        mlp = Seq(Linear((2 * in_c) + num_edge_features, out_c),
                  ReLU(),
                  Linear(out_c, out_c))
        print(mlp)
        self.NNConv = EdgeFeatsConv(in_c=in_c,
                                    out_c=out_c,
                                    nn=mlp,
                                    root_weight=root_weight
                                    )
        # self.NNConv = EdgeFeatsConv(in_c, out_c, mlp)
        # self.NNConv = geo_nn.GraphUNet(in_c, 4, out_c,depth=2, pool_ratios=0.5)
        self.bn = bn
        self.dropout = dropout
        if bn:
            self.batchNorm = nn.BatchNorm1d(out_c)
    def forward(self, data):
        x, edge_index, edge_attr = data
        if self.dropout:
            edge_index, edge_attr = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=0.1, force_undirected=True,
                                                     )
            # x = F.dropout(x, p=0.1)
        x = self.NNConv(x, edge_index, edge_attr)
        if self.bn:
            x = self.batchNorm(x)
        return x, edge_index, edge_attr

class EdgeFeatsConvMultNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(EdgeFeatsConvMultNN, self).__init__()
        root_weight = opts.root_weight
        num_edge_features = dataset.num_edge_features
        # num_node_features = dataset.num_node_features
        mlp_nodes = Seq(Linear(2 * in_c, out_c),
                  ReLU(),
                  Linear(out_c, out_c))
        mlp_edges = nn.Sequential(
            nn.Linear(num_edge_features + in_c, out_c),
            nn.ReLU(),
            # nn.Linear(num_node_features,num_node_features)
        )
        # mlp_edges = nn.Sequential(
        #     nn.Linear(num_edge_features + in_c, 128),
        #     nn.ReLU(),
        #     nn.Linear(128 , 64),
        #     nn.ReLU(),
        #     nn.Linear(64, out_c),
        #     nn.ReLU(),
        # )
        print(mlp_nodes)
        print(mlp_edges)
        self.NNConv = EdgeFeatsConvMult(
            in_c=in_c,
            out_c=out_c,
            nn_nodes=mlp_nodes,
            nn_edges=mlp_edges,
            root_weight=root_weight
        )
        # self.NNConv = EdgeFeatsConv(in_c, out_c, mlp)
        # self.NNConv = geo_nn.GraphUNet(in_c, 4, out_c,depth=2, pool_ratios=0.5)
        self.bn = bn
        self.dropout = dropout
        if bn:
            self.batchNorm = nn.BatchNorm1d(out_c)
    def forward(self, data):
        x, edge_index, edge_attr = data
        if self.dropout:
            edge_index, edge_attr = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=0.1, force_undirected=True,
                                                     )
            # x = F.dropout(x, p=0.1)
        x = self.NNConv(x, edge_index, edge_attr)
        if self.bn:
            x = self.batchNorm(x)
        return x, edge_index, edge_attr

class Net(torch.nn.Module):

    def __init__(self, dataset, opts):
        """
        model default: EdgeConv
        """
        super(Net, self).__init__()
        self.opts = opts
        layers = opts.layers
        layers_MLP = opts.layers_MLP
        self.layers = layers
        self.gnn = layers != [0]
        model = opts.model
        model = model.lower()
        layer_used = "edgeconv"
        if model == "edgeconv":
            layer_used = myNNConv
        elif model == "edgefeatsconv":
            layer_used = EdgeFeatsConvNN
        elif model == "edgefeatsconvmult":
            layer_used = EdgeFeatsConvMultNN
        else:
            print("Model {} not implemented".format(model))
            exit()

        # model = [myNNConv2(dataset.num_node_features, dataset.num_classes),
        #          ]
        if self.gnn:
            model = [
                EdgeFeatsConvNN(dataset.num_node_features, layers[0], dataset=dataset, opts=opts,),
                myNNConv(layers[0], layers[1], dataset=dataset, opts=opts,),
                    ]

            for i in range(2, len(layers)):

                model = model + [
                    layer_used(layers[i - 1], layers[i], dataset=dataset, opts=opts,),
                ]


            # model = model + [
            #     layer_used(layers[-1], dataset.num_classes, bn=False, dropout=False, dataset=dataset, opts=opts,)
            # ]
            self.model = nn.Sequential(*model)
        else:
            layers = [dataset.num_node_features]
        if opts.GL_type == "abs_diff":
            if layers_MLP != [0]:
                model2 = [nn.Linear(layers[-1], layers_MLP[0]),
                                nn.BatchNorm1d(layers_MLP[0]),
                                nn.ReLU(True),
                    ]
                
                for i in range(1, len(layers_MLP)):
                    model2 = model2 + [nn.Linear(layers_MLP[i-1], layers_MLP[i]),
                            nn.BatchNorm1d(layers_MLP[i]),
                            nn.ReLU(True),
                            # nn.Dropout(dropout)
                            ]
                model2 = model2 + [
                        nn.Linear(layers_MLP[-1], dataset.num_classes),
                        # nn.Softmax()
                ]
            else:
                 model2 = [nn.Linear(layers[-1], dataset.num_classes)]
            self.model2 = nn.Sequential(*model2)
        elif opts.GL_type == "bilinear":
            self.w_ = nn.Linear(layers[-1], layers[-1])
            self.bilinear = nn.Bilinear(layers[-1], layers[-1], dataset.num_classes)
        self.num_params = 0
        for param in self.parameters():
            self.num_params += param.numel()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch_ = data.batch
        edge_attr = data.edge_attr
        # print(edge_index)
        # print(edge_index.size())
        # x, edge_index, edge_attr = self.model([x, edge_index, edge_attr])
        if self.gnn:
            x, edge_index, edge_attr = self.model([x, edge_index, edge_attr])

        num_graphs = batch_.unique().size()[0]
        graphs = {i:[] for i in range(num_graphs)}
        g_all = []
        # print(x.size())
        for i, x_i in enumerate(x):
            batch_n = int(tensor_to_numpy(batch_[i]))
            graphs[batch_n].append(x_i)
        for i, g in graphs.items():
            g_ = torch.stack(g)
            size_g = len(g)
            # print(g_.size())
            if self.opts.GL_type == "abs_diff":
                g_2 = torch.cat([torch.abs(g_ - g[z]) for z in range(size_g)])
                # print(g_2.size())
                # exit(   )
                graphs[i] = g_2
            elif self.opts.GL_type == "bilinear":
                g_w = self.w_(g_)
                g_all.append(g_w.repeat(size_g, 1))
                g_2 = torch.cat([g[z].repeat(size_g,1) for z in range(size_g)])
                # ls.append(g_2)
                # g_2 = self.bilinear(g_all, g_2)
                # # print(g_2.size())
                # # exit(   )
                graphs[i] = g_2
        # [print(x.size()) for _,x in graphs.items()]
        all_x = torch.cat([ x for _,x in graphs.items()])
        # print(all_x.size())
        if self.opts.GL_type == "abs_diff":
            x = self.model2(all_x)
        elif self.opts.GL_type == "bilinear":
            g_all = torch.cat(g_all)
            x = self.bilinear(all_x, g_all )
        # print(x.size())
        # exit()
        del graphs
        del g_all
        return F.log_softmax(x, dim=1)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        try:
            nn.init.xavier_normal_(m.weight.data)
        except:
            print("object has no attribute 'weight'")
        # init.constant(m.bias.data, 0.0)
