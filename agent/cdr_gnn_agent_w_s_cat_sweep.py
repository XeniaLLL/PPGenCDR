import argparse
from pathlib import Path
import sys
DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME.parent))
import yaml

from rec_agent import *
import time
from utils.model_utils import plot_loss
import torch.nn.functional as F
from model.cfgan import GeneratorConEmb


class DiscriminatorConEmbOT(nn.Module):
    def __init__(self, itemCount, userCount, emb_size):
        super(DiscriminatorConEmbOT, self).__init__()

        self.user_embeddings = nn.Embedding(userCount, emb_size)
        self.user_embeddings.weight.data = torch.from_numpy(
            np.random.normal(0, 0.01, size=[userCount, emb_size])).float()

        self.encoder = nn.Sequential(
            nn.Linear(itemCount + emb_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, itemCount),
        )

    def forward(self, data, batch_users):
        rating_embeddings = data  # sself.rating_encoder(data)
        con_user_embeddings = self.user_embeddings(batch_users)
        data_c = torch.cat((rating_embeddings, con_user_embeddings), 1)
        latent = self.encoder(data_c)
        latent_act = F.relu(latent)
        result = self.decoder(latent_act)
        return result, latent


from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp


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
    adj = kneighbors_graph(features, knn_size, metric=knn_metric, include_self=True)
    adj_norm = normalize_sparse_adj(adj)
    if sparse_init_adj:
        adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
    else:
        adj_norm = torch.Tensor(adj_norm.todense())
    return adj_norm


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
        support = torch.matmul(input, self.weight)
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
        self.gat = GraphAttentionLayer(self.input_size, self.input_size, dropout=0.5)  # note　收敛很慢　暂时不考虑
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
            # agg_embed = self.gat(agg_embed, A_)
            if msg_dropout:
                agg_embed = self.dropout(agg_embed)
            embeds.append(agg_embed)
        embs = torch.stack(embeds, dim=1)
        return embs


class PPGCDR(nn.Module):
    def __init__(self, num_users, num_items_s, num_items_t, emb_size, knn_size, pool_method='mean',
                 K_negs=10, negative_sample_method='rns'):
        super(PPGCDR, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, emb_size)
        self.user_embeddings.weight.data = torch.from_numpy(
            np.random.normal(0, 0.01, size=[num_users, emb_size])).float()

        self.items_t_embeddings = nn.Embedding(num_items_t, emb_size)
        self.user_embeddings.weight.data = torch.from_numpy(
            np.random.normal(0, 0.01, size=[num_items_t, emb_size])).float()
        self.num_items_t = num_items_t
        # # note another implementation
        # initializer = nn.init.xavier_uniform_
        # self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        # self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))
        self.knn_size = knn_size
        self.pool = pool_method
        self.negative_sample_method = negative_sample_method
        self.K_negs = K_negs

        self.encoder_s = nn.Sequential(
            nn.Linear(num_items_s + emb_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
        )

        self.encoder_t = nn.Sequential(
            nn.Linear(num_items_t + emb_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
        )
        self.gnn_ui = GNN(emb_size, step=2, batch_norm=True, edge_dropout_rate=0.5, dropout=0.5)
        self.gnn_uu = GNN(emb_size, step=2, batch_norm=True, edge_dropout_rate=0.5, dropout=0.5)

        self.user_align = nn.Sequential(
            nn.Linear(emb_size * 2, emb_size),
            nn.Softmax(dim=1),
            nn.Linear(emb_size, emb_size)
        )  # note 用来拼接不同的user embedding信息

        # self.get_rating = nn.Sequential(
        #     nn.Linear(emb_size,)
        # )

        # self.gnn = GNN(hidden_size=256, step=2)
        self.SRP = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, num_items_s),
        )  # source rating predictor

        self.TRP = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, num_items_t),
        )

    def get_sparse_interaction_graph(self, batch_ratings):
        device = batch_ratings.device
        num_batch_users, num_items = batch_ratings.shape
        large_size = num_batch_users + num_items
        R = torch.zeros(large_size, large_size)
        R[:num_batch_users, num_batch_users:] = batch_ratings
        R[num_batch_users:, :num_batch_users] = batch_ratings.T
        D = R.sum(1).float()
        D[D == 0.] = 1.

        D_sqrt = torch.sqrt(D).unsqueeze(0)
        R = R / D_sqrt
        R = R / D_sqrt.t()
        graph = R.to(device)
        # index = R.nonzero()
        # data= R[R>= 1e-9]
        #
        # assert len(index)== len(data)
        #
        # # more for sparse matrix
        # graph= torch.sparse.FloatTensor(index.t(), data, torch.Size([large_size, large_size]))
        # graph=graph.coalesce()
        return graph

    def get_user_relation_graph(self, batch_ratings):
        device = batch_ratings.device
        batch_ratings = batch_ratings.cpu().detach().numpy()
        graph = gen_init_graph(batch_ratings, knn_size=self.knn_size)
        graph = graph.to(device)
        batch_ratings = torch.from_numpy(batch_ratings).to(device)
        return graph

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def bpr_loss(self, user_gnn_emb, item_gnn_emb):
        batch_size = user_gnn_emb.shape[0]
        u_e = self.pooling(user_gnn_emb)
        i_e = self.pooling(item_gnn_emb)

        pred_scores = torch.sum(torch.mul(u_e, i_e), axis=1)
        pass

    def negative_sampling(self, user_gcn_emb, item_gcn_item, user, neg_candidates, pos_items):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_item[pos_items]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        '''positive mixing: 插值混合--> 将正样本注入负样本中产生hard negative samples candidates'''
        seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)
        n_e = item_gcn_item[neg_candidates]
        n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e

        '''hop mixing: 利用选择策略得到上面候选的唯一样本,然后pooling,从而生成假的富含信息的负样本'''
        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs, n_hops+1/ n_layers+1]
        indices = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])
        return neg_items_emb_[[[i] for i in range(batch_size)], range(neg_items_emb_.shape[1]), indices, :]

    def forward(self, batch_user, batch_user_ratings_s, batch_user_ratings_t):
        '''
        version 2 s 的信息在s 的图上传一波,然后和t的信息拼接,基于www19 Wenqi Fan 做MLP prediction
        careful 验证合理性
        :param batch_user:
        :param batch_user_ratings_s:
        :param batch_user_ratings_t:
        :return:
        '''
        n_batch_users = batch_user.shape[0]
        batch_users_embed = self.user_embeddings(batch_user)

        all_items = torch.LongTensor(np.array(range(self.num_items_t))).to(batch_user.device)
        all_items_t_embed = self.items_t_embeddings(all_items)

        # todo add the graph here
        # # build ui graph
        A_hat_ui = self.get_sparse_interaction_graph(batch_user_ratings_t)
        A_hat_uu = self.get_user_relation_graph(batch_user_ratings_s)  # 学长的意思是这weight当作约束放在下面用
        batch_ui_embed_t = torch.cat((batch_users_embed, all_items_t_embed), dim=0)

        g_hidden_t = self.gnn_ui(batch_ui_embed_t, A_hat_ui)
        g_hidden_t_u, g_hidden_t_i = g_hidden_t[:n_batch_users, :], g_hidden_t[n_batch_users:, :]
        g_hidden_t_u = self.pooling(g_hidden_t_u)
        g_hidden_t_i = self.pooling(g_hidden_t_i)  # TODO pooling 内容和目的

        g_hidden_s_u = self.gnn_uu(batch_users_embed, A_hat_uu)
        g_hidden_s_u = self.pooling(g_hidden_s_u)

        g_hidden_u = self.user_align(torch.cat((g_hidden_s_u, g_hidden_t_u), dim=1))

        # g_hidden_u = g_hidden_t_u
        ratings = torch.matmul(g_hidden_u, g_hidden_t_i.T)

        # repeat_g_hidden_t_u = g_hidden_t_u.repeat(n_batch_users, 1, 1) #
        # norm_diff_matrix = repeat_g_hidden_t_u - g_hidden_t_u.unsqueeze(1)
        # norm = torch.matmul(norm_diff_matrix, norm_diff_matrix.transpose(2, 1))
        # loss_reg_user = -torch.mul(norm, A_hat_uu).mean()

        # diag_u = torch.mul(torch.diag(torch.ones(n_batch_users)), g_hidden_t_u)
        # diff = torch.norm(g_hidden_t_u - diag_u, p=1)
        # loss_reg_user = -A_hat_uu * diff.T  # careful todo check the shape

        # g_hidden_s = self.gnn(batch_user_ratings_s, A_hat_uu)
        # ratings = torch.cat((g_hidden_s, g_hidden_t), dim=1)

        # # # g_hidden_s, g_hidden_t = g_hidden.chunk(2, 1)  # 从dim=1 分出2个chunk
        # # preds_s = self.SRP(g_hidden_s)
        # preds_t = self.TRP(ratings)

        return (ratings,)  # loss_reg_user


class ETLGNNAgent(RecAgent):
    def __init__(self, args):
        super(ETLGNNAgent, self).__init__(args)
        self.model_s = GeneratorConEmb(self.num_items_s, self.num_users, args.emb_size)
        load_model(self.model_s, args.pp_G_s_model_path, strict=False)  # careful check it
        self.model_s.to(args.device)
        for param in self.model_s.parameters():
            param.requires_grad = False

        # model init
        self.model = PPGCDR(self.num_users, self.num_items_s, self.num_items_t, emb_size=args.emb_size,
                            knn_size=args.knn_size)
        self.model.to(args.device)
        self.lr = args.lr

        # self.opt = dp_optimizer.AdamDP(
        #     max_per_sample_grad_norm=1.0,  # max_per_sample_grad_norm,
        #     noise_multiplier=1.07,  # noise_multiplier,
        #     batch_size=self.batch_size,
        #     params=self.model.parameters(),
        #     lr=self.lr,
        #     betas=(0.5, 0.999),
        #     weight_decay=0.0001
        # )

        self.opt = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay
        )

        pos_weight = FloatTensor([args.pos_weight])  # , device=device)
        pos_weight = pos_weight.to(args.device)
        self.L_w_bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        self.L_bce = nn.BCEWithLogitsLoss(reduction="none")

        # self.user_ratings_s=pickle.load(open('fake_dp_False_ratings.pkl', 'rb'))
        # self.user_ratings_s=pickle.load(open('fake_dp_False_w_ot_ratings_minus_rec_g.pkl', 'rb'))
        # self.user_ratings_s=pickle.load(open('fake_dp_False_w_ot_ratings_no_rec_g.pkl', 'rb'))
        # self.user_ratings_s=pickle.load(open('fake_dp_False_w_ot_ratings_plus_rec_g.pkl', 'rb'))
        # self.user_ratings_s=pickle.load(open('fake_dp_True_w_ot_ratings_minus_rec_g.pkl', 'rb'))
        # self.user_ratings_s=pickle.load(open('fake_dp_True_w_ot_ratings_minus_rec_g_2*_1e-8.pkl', 'rb'))
        # self.user_ratings_s = pickle.load(open('fake_dp_True_x_w_ot_ratings_minus_rec_g.pkl', 'rb'))
        # self.user_ratings_s = pickle.load(open('fake_dp_True_x_noise_126_w_ot_ratings_minus_rec_g.pkl', 'rb'))
        # self.user_ratings_s = pickle.load(open('fake_dp_True_x_noise_2000_w_ot_ratings_minus_rec_g.pkl', 'rb'))
        # pass

    def fit(self, n_epochs, n_epochs_per_eval, topK_list=[], use_pretrain=False, use_model_tag='best_hr_s_pretrain',
            model_path=None, strict=False):
        if use_pretrain:
            load_model(self.model,
                       ((self.logPath / use_model_tag) if model_path is None else (model_path / use_model_tag)),
                       strict=strict)

        best_hr_t, best_ndcg_t, best_mrr_t = 0., 0., 0.  # Careful todo 这里用WANDB 维护
        best_hr_s, best_ndcg_s, best_mrr_s = 0., 0., 0.
        val_hr_s_list, val_ndcg_s_list, val_mrr_s_list = [], [], []
        val_hr_t_list, val_ndcg_t_list, val_mrr_t_list = [], [], []
        # loss computation list

        loss_list = []
        # loss_s_list = []
        loss_t_list = []

        for e in range(n_epochs):
            epoch_loss, epoch_loss_t, epoch_time = self.train_one_epoch(e)

            loss_list.append(epoch_loss)
            loss_t_list.append(epoch_loss_t)

            if e % n_epochs_per_eval == 0:
                self.model.eval()
                avg_hr_t, avg_ndcg_t, avg_mrr_t = test_process({'gen_s': self.model_s,
                                                                'model_t': self.model},
                                                               self.train_loader,
                                                               self.feed_data,
                                                               self.device,
                                                               topK=topK_list[1],
                                                               dr_target="cdr")
                wandb.log({
                    'val_hr_t': avg_hr_t,
                    'val_ndcg_t': avg_ndcg_t,
                    'val_mrr_t': avg_mrr_t,

                })
                val_hr_t_list.append(avg_hr_t)
                val_ndcg_t_list.append(avg_ndcg_t)
                val_mrr_t_list.append(avg_mrr_t)

                val_log_str = f'val: target: hr: {avg_hr_t}, ndcg: {avg_ndcg_t}, mrr: {avg_mrr_t}'
                print(val_log_str)
                with (self.logPath / 'tmp.txt').open('a') as fw:
                    fw.write(val_log_str)

                if avg_hr_t > best_hr_t:
                    torch.save(self.model.state_dict(), (self.logPath / 'best_hr_t.pkl'))
                    best_hr_t = avg_hr_t

                if avg_ndcg_t > best_ndcg_t:
                    torch.save(self.model.state_dict(), (self.logPath / 'best_ndcg_t.pkl'))
                    best_ndcg_t = avg_ndcg_t
                if avg_mrr_t > best_mrr_t:
                    torch.save(self.model.state_dict(), (self.logPath / 'best_mrr_t.pkl'))
                    best_mrr_t = avg_mrr_t
        loss_dict = {
            # 's_loss': loss_s_list,
            't_loss': loss_t_list,
            'loss': loss_list
        }
        plot_loss(loss_dict, self.logPath, loss_fig_title="etl_gnn_agent_loss")

    def testing(self, topK_list=[]):
        self.model.eval()
        print("Testing ...")
        model_best_hr = torch.load((self.logPath / 'best_hr_t.pkl'))
        model_best_ndcg = torch.load((self.logPath / 'best_ndcg_t.pkl'))
        model_best_mrr = torch.load((self.logPath / 'best_mrr_t.pkl'))

        for topK in topK_list:
            self.model.load_state_dict(model_best_hr)
            test_hr_t, _, _ = test_process({'gen_s': self.model_s,
                                            'model_t': self.model},
                                           self.train_loader,
                                           self.feed_data,
                                           self.device,
                                           topK=topK,
                                           dr_target="cdr", mode="test")

            self.model.load_state_dict(model_best_ndcg)
            _, test_ndcg_t, _ = test_process({'gen_s': self.model_s,
                                              'model_t': self.model},
                                             self.train_loader,
                                             self.feed_data,
                                             self.device,
                                             topK=topK,
                                             dr_target="cdr", mode="test")
            self.model.load_state_dict(model_best_mrr)
            _, _, test_mrr_t = test_process({'gen_s': self.model_s,
                                             'model_t': self.model},
                                            self.train_loader,
                                            self.feed_data,
                                            self.device,
                                            topK=topK,
                                            dr_target="cdr", mode="test")

            test_los_str = f"Test TopK:{topK}  target: hr: {test_hr_t}, ndcg: {test_ndcg_t} mrr: {test_mrr_t}"
            print(test_los_str)
            with (self.logPath / 'tmp.txt').open('a') as fw:
                fw.write(test_los_str)

    def train_one_epoch(self, epoch):
        self.model.train()

        batch_loss_list = []
        # batch_loss_reg_uu_list = []
        batch_loss_t_list = []
        epoch_time = 0.

        for batch_idx, data in enumerate(self.train_loader):
            data = data.reshape([-1])
            self.opt.zero_grad()
            batch_user = data.to(self.device)
            batch_user_rating_s = self.model_s(batch_user)[0]
            # batch_user_rating_s = self.user_ratings_s[data].to(self.device)
            batch_user_rating_t = self.user_ratings_t[data].to(self.device)
            time1 = time.time()
            preds_t = self.model(batch_user,
                                 batch_user_rating_s,
                                 batch_user_rating_t)[0]
            time2 = time.time()
            epoch_time += time2 - time1
            # loss_s = self.L_bce(preds_s, batch_user_rating_s).sum()
            loss_t = self.L_bce(preds_t, batch_user_rating_t).sum()
            loss = loss_t
            batch_loss_list.append(loss.item())
            batch_loss_t_list.append(loss_t.item())
            # batch_loss_reg_uu_list.append(loss_reg_uu.item())
            loss.backward()
            self.opt.step()

        # epoch_loss_reg_uu = np.mean(batch_loss_reg_uu_list)
        epoch_loss_t = np.mean(batch_loss_t_list)
        epoch_loss = np.mean(batch_loss_list)

        train_log_str = f"epoch: {epoch}, epoch loss: {epoch_loss},  t loss: {epoch_loss_t} "
        print(train_log_str)

        with (self.logPath / "tmp.txt").open('a') as fw:
            fw.write(train_log_str)

        return epoch_loss, epoch_loss_t, epoch_time


def main():
    with wandb.init(magic=True):
        wandb_config = wandb.config
        with Path('../config/etl_gnn_param_w_s.yaml').open("r") as f:
            config = yaml.safe_load(f)
        config.update(wandb_config)
        args = argparse.Namespace(**config)
        print(args)
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        args.device = DEVICE
        agent = ETLGNNAgent(args)
        agent.fit(args.epochs, 1, [5, 10])
        agent.testing([5, 10])
        print(args)


if __name__ == '__main__':
    from config_argparser import ConfigArgumentParser, YamlConfigAction
    import os
    import wandb

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    sweep_config = yaml.safe_load(Path("../config/cdr_gnn_cat_sweep.yml").open("r"))

    sweep_id = wandb.sweep(sweep_config, project='cat_gnn_w_s_cat_2')
    wandb.agent(sweep_id, function=main, count=300)

    # parser = ConfigArgumentParser('test')
    # parser.add_argument("--config", action=YamlConfigAction, default=['../config/etl_gnn_param_w_s.yaml'])
    # args = parser.parse_args()
    # print(args)
    #
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # args.device = DEVICE
    # agent = ETLGNNAgent(args)
    # agent.fit(args.epochs, 1, [5, 10])
    # agent.testing([5, 10])
    # print(args)
