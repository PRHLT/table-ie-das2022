import torch.nn.functional as F
from torch.nn.modules import dropout
import torch_geometric.nn as geo_nn
import torch
from torch import nn
import torch_geometric as tg
# from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.utils import to_undirected
from models.pna import PNAConvSimple
from torch_geometric.utils import degree
try:
    from models.nnconv import EdgeFeatsConv, EdgeFeatsConvMult, ECNConv, EdgeConv2
except:
    from .nnconv import EdgeFeatsConv, EdgeFeatsConvMult, ECNConv, EdgeConv2
from models.operations import Mish

# act_func = nn.ReLU
act_func = Mish


class GATv2ConvNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(GATv2ConvNN, self).__init__()
        root_weight = opts.root_weight
        num_edge_features = dataset.num_edge_features
        self.NNConv = geo_nn.GATv2Conv(in_channels=in_c,
                                    out_channels=out_c,
                                    dropout=dropout,
                                    heads=1
                                    )
        self.bn = bn
        self.dropout = dropout
        if bn:
            self.batchNorm = nn.BatchNorm1d(out_c)
    def forward(self, data):
        x, edge_index, edge_attr = data
        # if self.dropout:
        #     edge_index, edge_attr = tg.utils.dropout_adj(edge_index, edge_attr,
        #                                              p=0.1, force_undirected=True,
        #                                              )
            # x = F.dropout(x, p=0.1)
        x = self.NNConv(x, edge_index)
        if self.bn:
            x = self.batchNorm(x)
        return x, edge_index, edge_attr

class myNNConv(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=0.5, dataset=None, opts={}):
        super(myNNConv, self).__init__()
        mlp = Seq(Linear(2 * in_c, out_c),
                  nn.Dropout(opts.mlp_do),
                  act_func(),
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
        if self.dropout > 0:
            edge_index, edge_attr = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=self.dropout, force_undirected=True,
                                                     )
        x = self.NNConv(x, edge_index,)
        if self.bn:
            x = self.batchNorm(x)
        return x, edge_index, edge_attr

class myNNConv2(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(myNNConv2, self).__init__()
        mlp = Seq(Linear(2 * in_c, out_c),
                  act_func(),
                  Linear(out_c, out_c))
        self.k = opts.knn
        self.dynamic = self.k > 0
        self.NNConv = EdgeConv2(mlp)
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
                  act_func(),
                  Linear(out_c, out_c))
        print(mlp)
        self.NNConv = EdgeFeatsConv(in_c=in_c,
                                    out_c=out_c,
                                    nn=mlp,
                                    root_weight=root_weight
                                    )
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

class PNANN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=False, dropout=False, dataset=None, opts={}):
        super(PNANN, self).__init__()
        
        self.NNConv = PNAConvSimple(in_c,
                                    out_c,
                                    dataset,
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
                  act_func(),
                  Linear(out_c, out_c))
        mlp_edges = nn.Sequential(
            nn.Linear(num_edge_features + in_c, out_c),
            act_func(),
            # nn.Linear(num_node_features,num_node_features)
        )

        print(mlp_nodes)
        print(mlp_edges)
        self.NNConv = EdgeFeatsConvMult(
            in_c=in_c,
            out_c=out_c,
            nn_nodes=mlp_nodes,
            nn_edges=mlp_edges,
            root_weight=root_weight
        )

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

class GatConvNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(GatConvNN, self).__init__()
        root_weight = opts.root_weight
        num_edge_features = dataset.num_edge_features

        self.NNConv = tg.nn.GATConv(in_channels =in_c,
                                    out_channels =out_c,
                                    heads=4,
                                    concat=False
                                    )
    def forward(self, data):
        x, edge_index, edge_attr = data
        x = self.NNConv(x, edge_index)
        return x, edge_index, edge_attr

class ECNConvNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(ECNConvNN, self).__init__()
        root_weight = opts.root_weight
        num_edge_features = dataset.num_edge_features
        self.NNConv = ECNConv(in_channels =in_c,
                                    out_channels =out_c,
                                    num_edge_features=num_edge_features,
                                    # concat=False
                                    )
    def forward(self, data):
        x, edge_index, edge_attr = data
        x = self.NNConv(x, edge_index, edge_attr)
        return x, edge_index, edge_attr

class TransformerNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=False, dataset=None, opts={}):
        super(TransformerNN, self).__init__()
        root_weight = opts.root_weight
        num_edge_features = dataset.num_edge_features
        self.k = opts.knn
        self.dynamic = self.k > 0
        self.bn = bn
        self.dropout = dropout
        self.NNConv = geo_nn.TransformerConv(in_channels =in_c,
                                    out_channels =out_c,
                                    heads=4,
                                    edge_dim = num_edge_features,
                                    concat=False,
                                    dropout=0.5
                                    )
        if bn:
            # self.batchNorm = geo_nn.GraphNorm(out_c)
            self.batchNorm = geo_nn.BatchNorm(out_c)
        
    def forward(self, data):
        x, edge_index, edge_attr = data
        if self.dynamic:
            print("dynamic, ", self.k)
            edge_index = geo_nn.knn_graph(x, self.k, 
            # batch=batch, 
            loop=False, cosine=False)
            edge_index = to_undirected(edge_index)
        # if self.dropout:
        #     edge_index, edge_attr = tg.utils.dropout_adj(edge_index, edge_attr,
        #                                              p=0.1, force_undirected=True,
        #                                              )
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
        model = opts.model
        model = model.lower()
        # layer_used = "edgeconv"
        # model = "ecn"
        # layer_used = opts.model
        # model = "pna"
        if model == "edgeconv":
            layer_used = myNNConv
        elif model == "edgeconv2":
            layer_used = myNNConv2
        elif model == "edgefeatsconv":
            layer_used = EdgeFeatsConvNN
        elif model == "edgefeatsconvmult":
            layer_used = EdgeFeatsConvMultNN
        elif model == "pna":
            layer_used = PNANN
        elif model == "gat":
            layer_used = GatConvNN
        elif model == "ecn":
            layer_used = ECNConvNN
        elif model == "transformer":
            layer_used = TransformerNN
        elif model == "gatv2":
            layer_used = GATv2ConvNN
        else:
            print("Model {} not implemented".format(model))
            exit()

        print("Using {} layers".format(layer_used))
        # print("dataset.num_node_features", dataset.num_node_features)
        dropout_adj = opts.do_adj
        self.use_GNN = True
        if layers and layers[0] != 0:
            model = [
                layer_used(dataset.num_node_features, layers[0], dataset=dataset, opts=opts, dropout=dropout_adj),
                layer_used(layers[0], layers[1], dataset=dataset, opts=opts, dropout=dropout_adj),
                    ]

            for i in range(2, len(layers)):

                model = model + [
                    layer_used(layers[i - 1], layers[i], dataset=dataset, opts=opts, dropout=dropout_adj,),
                ]
            if self.opts.classify != "EDGES":
                if self.opts.g_loss == "NLL":
                    model = model + [
                        layer_used(layers[-1], dataset.num_classes, bn=False, dropout=dropout_adj, dataset=dataset, opts=opts,)
                    ]
                else:
                    model = model + [
                        layer_used(layers[-1], 1, bn=False, dropout=dropout_adj, dataset=dataset, opts=opts)
                    ]
            else:
                len_feats = dataset.num_node_features + layers[-1] + dataset.num_edge_features
                self.mlp_edges = self.get_mlp_edges(len_feats, dataset, layers_MLP, opts)
            if not opts.use_residual:
                self.model = nn.Sequential(*model)
            else:
                # model = [x.to("cuda") for x in model]
                self.model = nn.ModuleList(model)
        else:
            self.use_GNN = False
            len_feats = dataset.num_node_features + dataset.num_edge_features
            self.mlp_edges = self.get_mlp_edges(len_feats, dataset, layers_MLP, opts)
        
        
        self.num_params = 0
        for param in self.parameters():
            self.num_params += param.numel()

    def get_mlp_edges(self, len_feats, dataset, layers, opts):
        if layers == [0]:
            model = [nn.Linear((len_feats), 2)]
        else:
            if len(layers) == 1:
                model = [nn.Linear(len_feats, layers[0]),
                nn.BatchNorm1d(layers[0]),
                nn.Linear(layers[0], 2)]
            else:
                model = [nn.Linear((len_feats), layers[0]),
                         nn.BatchNorm1d(layers[0]),
                         nn.ReLU(True),
                         nn.Dropout(0.5)
                         ]
                for i in range(1, len(layers)):
                    model = model + [nn.Linear(layers[i-1], layers[i]),
                         nn.BatchNorm1d(layers[i]),
                         nn.ReLU(True),
                         nn.Dropout(0.5)
                         ]
                model = model + [
                        nn.Linear(layers[-1], 2),
                ]
       
        mlp_edges = nn.Sequential(*model)
        return mlp_edges
            
    def calc_feats_edges(self, x, edge_index, edge_attr=None, x_orig=None):
        s = x[edge_index[0]]
        d = x[edge_index[1]]
        edge_info = (s-d).abs()
        if edge_attr is not None and x_orig is None:
            edge_info = torch.cat([edge_info, edge_attr], dim=1)
        elif edge_attr is not None and x_orig is not None:
            s_orig = x_orig[edge_index[0]]
            d_orig = x_orig[edge_index[1]]
            edge_info_orig = (s_orig-d_orig).abs()
            edge_info = torch.cat([edge_info, edge_attr, edge_info_orig], dim=1)
        feats = self.mlp_edges(edge_info)
        return feats

    def calc_feats_edges_nonGNN(self,edge_index, edge_attr=None, x_orig=None):
        s_orig = x_orig[edge_index[0]]
        d_orig = x_orig[edge_index[1]]
        edge_info_orig = (s_orig-d_orig).abs()
        edge_info = torch.cat([edge_attr, edge_info_orig], dim=1)
        feats = self.mlp_edges(edge_info)
        return feats

    def forward_residual(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr
        res = []
        for i, net in enumerate(self.model):
            if i % 2 == 0 and i > 0:
                x += res[i-2]
            # print(x)
            # print(edge_index)
            # print(edge_attr)
            x, edge_index, edge_attr = self.model[i]([x, edge_index, edge_attr])
            res.append(x)
        del res
        return x, edge_index, edge_attr

    def forward(self, data):
        x_orig, edge_index_orig = data.x, data.edge_index
        edge_attr_orig = data.edge_attr
        if self.use_GNN:
            if not self.opts.use_residual:
                x, edge_index, edge_attr = self.model([x_orig, edge_index_orig, edge_attr_orig])
            else:
                x, edge_index, edge_attr  = self.forward_residual(data)
            if self.opts.classify == "EDGES":
                x = self.calc_feats_edges(x, edge_index_orig, edge_attr_orig, x_orig)
        else:
            x = self.calc_feats_edges_nonGNN(edge_index_orig, edge_attr_orig, x_orig)
        if self.opts.g_loss == "NLL":
            return F.log_softmax(x, dim=    1)
        else:
            return torch.squeeze(x)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        try:
            nn.init.xavier_normal_(m.weight.data)
        except:
            print("object has no attribute 'weight'")
        # init.constant(m.bias.data, 0.0)


# if __name__ == "__main__":
    