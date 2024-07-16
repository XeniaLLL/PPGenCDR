from import_torch import *

########################### note GNN generator as below
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph

def normalize_sparse_adj(mx):
    """Row-normalize sparse matrix: symmetric normalized Laplacian"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def gen_init_graph(features, knn_size, knn_metric='cosine', sparse_init_adj=False):
    '''

    :param features: the embedding matrix
    :param knn_size: n_neighbors per node
    :param knn_metric: metric for distance computation. default is cosine
    :param sparse_init_adj:
    :return:
    '''
    # todo add normalize_features elegantly
    # print('[ Using KNN-graph as input graph: {} ]'.format(knn_size))
    adj = kneighbors_graph(features, n_neighbors=knn_size, metric=knn_metric, include_self=True,
                           p=2)  # mode='distance',
    # careful 是否支持torch.tensor
    adj_norm = normalize_sparse_adj(adj)
    if sparse_init_adj:
        adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
    else:
        adj_norm = torch.Tensor(adj_norm.todense())
    return adj_norm

########################### note GNN model as below

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, graph_hops, dropout, batch_norm=False):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(GCNLayer(nfeat, nhid, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(GCNLayer(nhid, nhid, batch_norm=batch_norm))

        self.graph_encoders.append(GCNLayer(nhid, nclass, batch_norm=False))

    def forward(self, x, node_anchor_adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = encoder(x, node_anchor_adj)
            # x = F.relu(encoder(x, node_anchor_adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, node_anchor_adj)
        return x


class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(GCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.uniform_(
                self.bias))  # careful bias= torch.nn.xavier_uniform_ initialize is not allowed for one-dim bias
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None  # out_features == n_channels

    def forward(self, input, adj, batch_norm=True):
        support = torch.matmul(input, self.weight)  # careful 是谁乘以谁
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)  # note batch_size, channels, emb_size 这个shape才是对的
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class GNN(nn.Module):
    def __init__(self, hidden_size, step=1, batch_norm=False, edge_dropout_rate=0.5, dropout=0.5):
        # todo knn_size 改成和batch 有关的参数
        super(GNN, self).__init__()
        self.step = step
        self.input_size = hidden_size  # * 2
        # self.knn_size = knn_size
        self.batch_norm = batch_norm
        self.edge_dropout_rate = edge_dropout_rate
        self.gcn = GCNLayer(self.input_size, self.input_size, bias=True, batch_norm=batch_norm)
        # self.gcn= GraphAttentionLayer(self.input_size, self.input_size, dropout=0.5, alpha=0.2, concat=True)
        self.dropout = nn.Dropout(p=dropout)

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()  # note 针对sparse 会有的# of zeros todo 我这里没有用sparse 那一套,改为正常的size()/.shape
        random_tensor = rate
        random_tensor += torch.rand(noise_shape).tp(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, hidden, A, edge_dropout=False, msg_dropout=False):
        '''
        :param A:
        :param hidden:
        :param graph_mask: todo
        :return:
        '''
        embeds = [hidden]
        agg_embed = hidden

        for i in range(self.step):
            # update satellite
            A_ = self._sparse_dropout(A, self.edge_dropout_rate) if edge_dropout else A
            agg_embed = self.gcn(agg_embed, A_, self.batch_norm)
            # agg_embed = self.gcn(agg_embed, A_)
            if msg_dropout:
                agg_embed = self.dropout(agg_embed)
            embeds.append(agg_embed)
        embs = torch.stack(embeds, dim=1)
        return embs



class GraphSAGE(nn.Module):
    """https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/train_full.py"""

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        from dgl.nn.pytorch.conv import SAGEConv

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))  # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input1 = torch.matmul(h, self.a1)
        a_input2 = torch.matmul(h, self.a2)
        e = self.leakyrelu(a_input1 + a_input2.transpose(-1, -2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x