from pathlib import Path
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME.parent))
from rec_agent import *
from geomloss import SamplesLoss
from autodp.autodp_core import Mechanism
from autodp.transformer_zoo import Composition
from autodp import mechanism_zoo, transformer_zoo
from autodp import rdp_acct, rdp_bank
import time
from utils.model_utils import plot_loss, toggle_grad
from model.pid_optimizer import *
from model.random_queue import Random_queue
from model.ppgan import PPGAN, HeteroCDR
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import yaml
import argparse

class CDRGanGNNAgent(RecAgentchgLoader, RecAgentNegSample):
    def __init__(self, args):
        super(CDRGanGNNAgent, self).__init__(args)
        self.device = args.device
        self.is_PP = args.is_PP
        self.num_of_candidate_negative = args.num_of_candidate_negative
        self.dataset_name= args.dataset
        self.noise_multiplier= args.noise_multiplier_

        # note modeling source
        self.batch_size_model_t = args.batch_size_model_t
        self.batch_size_model_s = args.batch_size_model_s
        self.model_s = PPGAN(self.num_items_s, self.num_users, args.emb_size, args.is_PP, args.lr_g, args.lr_d, args.m,
                             args.dataset, self.noise_multiplier)
        # load_model(self.model_s, args.pp_G_s_model_path, strict=True)  # careful check it
        self.model_s.to(self.device)
        # for param in self.model_s.parameters():
        #     param.requires_grad = False

        # note modeling target
        # self.model_t = PPGCDR(self.num_users, self.num_items_s, self.num_items_t, emb_size=args.emb_size,
        #                       knn_size=args.knn_size, items_t_embed=self.items_t_embeddings, pool_method='mean',
        #                       lamda=args.ns_lamda,
        #                       gamma=args.ns_gamma, reg_uu=args.lamb_uu, K=args.num_of_candidate_negative)
        #
        self.model_t = HeteroCDR(self.num_users, num_items_s=self.num_items_s, num_items_t=self.num_items_t,
                                 emb_size=args.emb_size, dropout=args.dropout, is_sparse=False,
                                 dataset_name=args.dataset)
        self.model_t.to(self.device)

        # note learning rate
        self.lr_t = args.lr_t
        self.lr_g = args.lr_g
        self.lr_d = args.lr_d

        # self.lamb_uu = args.lamb_uu

        # note optimziers
        self.pv = args.pv
        self.iv = args.iv
        self.dv = args.dv
        self.i_buffer_factor = args.i_buffer_factor

        self.i_queue = Random_queue(
            self.batch_size_model_s * self.i_buffer_factor, self.batch_size_model_s
            # urgent todo i_buffer_factor 的物理含义
        )

        self.opt_g = torch.optim.RMSprop(
            params=self.model_s.generator_s.parameters(),
            lr=self.lr_g,
            alpha=0.99,
            eps=1e-8,
            weight_decay=args.weight_decay
        )

        self.opt_d = PID_RMSprop(
            params=self.model_s.discriminator_s.parameters(),
            lr=self.lr_d,
            eps=1e-8,
            vp=0.0001,
            vi=0.,
            vd=0.1
        )

        self.opt_t = torch.optim.Adam(
            params=self.model_t.parameters(),
            lr=self.lr_t,
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay
        )

        self.scheduler_t = torch.optim.lr_scheduler.StepLR(self.opt_t, step_size=20,
                                                           gamma=0.65)
        # note loss for model t
        # pos_weight = FloatTensor([args.pos_weight])  # , device=device)
        # pos_weight = pos_weight.to(args.device)
        # self.L_w_bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        # self.L_bce = nn.BCEWithLogitsLoss(reduction="none")

        self.loss_t_method = args.loss_t_method
        self.d_g_steps = args.d_g_steps
        self.d_steps = args.d_steps
        self.t_s_epochs = args.t_s_epochs
        self.s_epochs = args.s_epochs

        self.batch_size_model_t = args.batch_size_model_t
        self.batch_size_model_s = args.batch_size_model_s

        self.d_previous_user_embed = None
        self.d_previous_user_s_ratings = None

        self.one= torch.FloatTensor([1]).to(self.device)
        self.m_one= torch.FloatTensor([-1]).to(self.device)

        # # self.draw_heat_map(self.user_ratings_s[820:860, 512:550],"raw_real")
        # # self.R_s_10 = pickle.load(open("fake_ratings_10.pkl", 'rb')).to(self.device)
        # self.R_s_20 = pickle.load(open("fake_ratings_20.pkl", 'rb')).to(self.device)
        # # self.draw_heat_map(self.R_s_20.clone().cpu()[120:140, -10:],"generated_fake")
        # # self.draw_heat_map(self.R_s_20.clone().cpu()[220:240,  -10:],"generated_fake")
        # # self.draw_heat_map(self.R_s_20.clone().cpu()[820:860,   512:550],"generated_fake")
        # self.draw_heat_map(self.user_ratings_s[120:180, 512:550],"raw_real")
        # self.draw_heat_map(self.R_s_20.clone().cpu()[120:180,   512:550],"generated_fake")
        #
        # self.draw_heat_map(self.user_ratings_s[240:280, 512:550], "raw_real")
        # self.draw_heat_map(self.R_s_20.clone().cpu()[240:280, 512:550], "generated_fake")
        # # self.R_s_30 = pickle.load(open("fake_ratings_30.pkl", 'rb')).to(self.device)
        # # self.R_s_40 = pickle.load(open("fake_ratings_40.pkl", 'rb')).to(self.device)
        # note privacy analysis
        if args.noise_multiplier_ !=99:
        # self.privacy_analysis_search(args.epochs)
            self.privacy_analysis(args.epochs)

        pass


    def privacy_analysis_search(self, n_epochs=1):
        delta = 1e-5
        sigma = self.noise_multiplier

        n_steps = n_epochs * len(self.train_loader)
        # prob = 1 / len(self.train_loader)
        prob = 1 / (len(self.train_loader) / 2)
        rang_ = np.array(list(range(100, 500))) * 0.5

        for si in list(rang_):
            func = lambda x: rdp_bank.RDP_gaussian({"sigma": si}, x)
            acct = rdp_acct.anaRDPacct()
            acct.compose_subsampled_mechanism(func, prob, coeff=n_steps * self.batch_size)
            epsilon = acct.get_eps(delta=delta)  # note rdp 算出再转到 eps
            print("Privacy cost is: sigma={}, epsilon={}, delta={}".format(si, epsilon, delta))


    def privacy_analysis(self, n_epochs=1):
        # urgent todo
        delta = 1e-5
        sigma = self.noise_multiplier
        # print("noise_multiplier in privacy_analysis: ", noise_multiplier, sigma)

        n_steps = n_epochs * len(self.train_loader)
        # prob = 1 / len(self.train_loader)
        prob = 1 / (len(self.train_loader) / 3 * 2)
        func = lambda x: rdp_bank.RDP_gaussian({"sigma": sigma}, x)
        acct = rdp_acct.anaRDPacct()
        acct.compose_subsampled_mechanism(func, prob, coeff=n_steps * self.batch_size)
        epsilon = acct.get_eps(delta=delta)  # note rdp 算出再转到 eps
        print("Privacy cost is: sigma={}, epsilon={}, delta={}".format(sigma, epsilon, delta))
        wandb.log({
            'n_steps':n_steps,
            'sigma':sigma,
            'epsilon': epsilon,
            'delta':delta,
        })


    def draw_heat_map(self, val, title='todo'):
        '''

        :param val: narray is supposed
        :return:
        '''

        fig, ax = plt.subplots(figsize=(val.shape[0], val.shape[1]))
        plt.title(title, fontsize=18)
        ttl = ax.title
        ttl.set_position([0.5, 1.05])
        ax.set_xticks([])  # hide ticks for axis
        ax.set_yticks([])
        ax.axis("off")
        sns.heatmap(val, annot=None, fmt=".1f", cmap="RdYlGn", square=True, ax=ax, cbar=False)
        plt.show()
        plt.close()

    def train_model_s(self, epoch):
        self.model_s.train()
        self.model_t.eval()
        toggle_grad(self.model_t, False)
        # # user unfreezed generator
        # for param in self.model_s.generator_s.parameters():
        #     param.requires_grad = True

        if self.is_PP:
            self.model_s.apply_register_hook()

        # epoch_time = 0.
        batch_D_t_loss_list = []
        batch_G_t_loss_list = []
        time1 = time.time()
        self.change_loader_batch_size(self.batch_size_model_s)

        for batch_idx, data in enumerate(self.train_loader):

            data = data.reshape([-1])
            if (batch_idx) % (self.d_g_steps) < self.d_steps:
                toggle_grad(self.model_s.generator_s, False)
                toggle_grad(self.model_s.discriminator_s, True)
                self.model_s.generator_s.train()
                self.model_s.discriminator_s.train()
                self.opt_d.zero_grad()
                # self.opt_use_embed.zero_grad()

                batch_user_s_ratings = self.user_ratings_s[data].to(self.device)
                data = data.to(self.device)
                # user_embed = self.user_embeddings(data)
                user_embed = data
                d_loss, d_real, d_fake, w_distance = self.model_s.train_D_s(user_embed, batch_user_s_ratings)


                d_real.backward(self.one)
                d_fake.backward(self.m_one)

                iv_loss = torch.from_numpy(np.array([0.]))
                if self.iv > 0:
                    batch_user_embed_temp = user_embed.detach().cpu().numpy()
                    batch_user_s_ratings_sample_temp = batch_user_s_ratings.detach().cpu().numpy()
                    # d_real_recon_temp = d_real_recon.detach().cpu().numpy()
                    # d_fake_recon_temp= d_fake_recon.detach().cpu().numpy()
                    # d_real_latent_temp = d_real_latent.detach().cpu().numpy()
                    # d_fake_latent_temp = d_fake_latent.detach().cpu().numpy()
                    self.i_queue.set_data(batch_user_embed_temp, batch_user_s_ratings_sample_temp)

                    i_user_embed, i_user_s_ratings_sample = self.i_queue.get_data()
                    i_user_embed = torch.from_numpy(i_user_embed).to(self.device)
                    i_user_s_ratings_sample = torch.from_numpy(i_user_s_ratings_sample).to(
                        self.device)
                    iv_d_loss, iv_d_real_loss, iv_d_fake_loss, _ = self.model_s.train_D_s(i_user_embed, i_user_s_ratings_sample)
                    iv_loss = (iv_d_real_loss+ iv_d_fake_loss) * self.iv
                    iv_loss.backward()

                dv_loss = torch.from_numpy(np.array([0.]))
                if self.dv > 0 and batch_idx > 0:
                    if self.d_previous_user_embed is None:
                        self.d_previous_user_embed = user_embed
                        self.d_previous_user_s_ratings = batch_user_s_ratings

                    else:
                        dv_loss_previous,dv_loss_real_previous ,dv_loss_fake_previous,_= self.model_s.train_D_s(self.d_previous_user_embed,
                                                                  self.d_previous_user_s_ratings)
                        dv_loss_previous= dv_loss_real_previous+ dv_loss_fake_previous
                        dv_loss_current,dv_loss_real_current ,dv_loss_fake_current,_ = self.model_s.train_D_s(user_embed, batch_user_s_ratings)
                        dv_loss_current =dv_loss_real_current+dv_loss_fake_current
                        dv_loss = (dv_loss_current - dv_loss_previous) * self.dv
                        dv_loss.backward()
                        self.d_previous_user_embed = user_embed
                        self.d_previous_user_s_ratings = batch_user_s_ratings

                self.opt_d.step()
                # self.opt_use_embed.step()
                batch_D_t_loss_list.append(d_loss.item())
                toggle_grad(self.model_s.discriminator_s, False)
                # print("d ================")

            else:
                # print("g ================")
                toggle_grad(self.model_s.generator_s, True)
                toggle_grad(self.model_s.discriminator_s, False)
                self.model_s.generator_s.train()
                self.model_s.discriminator_s.train()
                self.opt_g.zero_grad()
                # self.opt_use_embed.zero_grad()
                batch_user_s_ratings = self.user_ratings_s[data].to(self.device)
                data = data.to(self.device)
                # user_embed = self.user_embeddings(data)
                user_embed = data
                g_loss = self.model_s.train_G_s(user_embed,
                                                batch_user_s_ratings)
                g_loss.backward(self.one)
                self.opt_g.step()
                # self.opt_use_embed.step()
                batch_G_t_loss_list.append(g_loss.item())

        time2 = time.time()
        epoch_d_loss = np.mean(batch_D_t_loss_list)
        epoch_g_loss = np.mean(batch_G_t_loss_list)

        train_log_str = f"Epoch: {epoch} train S, d loss: {epoch_d_loss} g loss: {epoch_g_loss}"
        print(train_log_str)

        with (self.logPath / "tmp.txt").open('a') as fw:
            fw.write(train_log_str)
        return epoch_d_loss, epoch_g_loss, (time2 - time1)

    def train_model_t(self, epoch):
        self.model_t.train()
        self.model_s.eval()
        # user freezed generator
        toggle_grad(self.model_s, False)
        toggle_grad(self.model_t, True)
        # toggle_grad(self.model_s.discriminator_s, False)
        # for param in self.model_s.generator_s.parameters():
        #     param.requires_grad = False

        batch_loss_t_list = []
        epoch_time = 0.
        self.change_loader_batch_size(self.batch_size_model_t)

        negative_samples_dict = self.negative_samples(domain='t', K=self.num_of_candidate_negative,
                                                      return_dict=True,
                                                      use_diff_neg_per_pos=True)  # 每个epoch进行一轮采样, 每个user 同一组的负样本

        for batch_idx, data in enumerate(self.train_loader):
            batch_neg_set_list = []
            batch_pos_set_list = []
            batch_user_set_list = []
            for user_idx in data.clone().detach():
                data_idx = str(user_idx[0].numpy())  # 把tensor 转成对应的字符串作为key 真丑
                user_positive_samples, user_negative_samples = negative_samples_dict[data_idx]
                user_positive_samples = torch.tensor(user_positive_samples)
                user_negative_samples = torch.tensor(
                    user_negative_samples).squeeze(1)  # todo negative sample for each one by one mapping
                # user_expand= user_idx * torch.ones_like(user_positive_samples)
                batch_pos_set_list.append(user_positive_samples)
                batch_neg_set_list.append(user_negative_samples)
                # batch_user_set_list.append(user_expand)

            batch_pos_set = batch_pos_set_list
            batch_neg_set = batch_neg_set_list
            # batch_pos_set = torch.cat(batch_pos_set_list, dim=0).to(self.device)
            # batch_neg_set = torch.cat(batch_neg_set_list, dim=0).to(self.device)
            # batch_user_set = torch.cat(batch_user_set_list, dim=0).to(self.device)
            data = data.reshape([-1])
            self.opt_t.zero_grad()
            batch_user = data.to(self.device)
            # batch_user_embed = self.user_embeddings(batch_user)
            batch_user_ratings_s = self.model_s(batch_user)[0]
            # batch_user_ratings_s = self.R_s_10[batch_user]
            # batch_user_ratings_s = self.R_s_20[batch_user]
            # batch_user_ratings_s = self.R_s_30[batch_user]
            # batch_user_ratings_s = self.R_s_40[batch_user]
            batch_user_ratings_t = self.user_ratings_t[data].to(self.device)

            time1 = time.time()
            loss_t = self.model_t.compute_loss(batch_user, batch_user_ratings_s,
                                               batch_user_ratings_t, batch_pos_set, batch_neg_set,
                                               loss_method=self.loss_t_method)  # urgent loss_method
            loss_t.backward()
            self.opt_t.step()
            time2 = time.time()
            epoch_time += time2 - time1
            # loss_t = loss_pred_t + self.lamb_uu * loss_reg_uu
            batch_loss_t_list.append(loss_t.item())

        # epoch_loss_reg_uu = np.mean(batch_loss_reg_uu_list)
        # epoch_loss_pred_t = np.mean(batch_loss_pred_t_list)
        epoch_loss_t = np.mean(batch_loss_t_list)
        train_log_str = f"epoch train T: {epoch}, epoch loss: {epoch_loss_t} "
        print(train_log_str)

        with (self.logPath / "tmp.txt").open('a') as fw:
            fw.write(train_log_str)
        return epoch_loss_t, epoch_time

    def fit(self, n_epochs, n_epochs_per_eval=1, topK_list=[], use_pretrain=False, use_model_tag='best_hr_s.pkl',
            model_path=None, strict=True, do_eval_model_s=True, do_eval_model_t=True):
        if use_pretrain:
            # model_dict = {
            #     's': self.model_s,
            #     't': self.model_t
            # }
            # load_model(model_dict,
            #            ((self.logPath / use_model_tag) if model_path is None else (model_path / use_model_tag)),
            #            strict=strict)
            load_model(self.model_s,
                       ((self.logPath / use_model_tag) if model_path is None else (model_path / use_model_tag)),
                       strict=strict)

        best_hr_t, best_ndcg_t, best_mrr_t = 0., 0., 0.
        best_hr_s, best_ndcg_s, best_mrr_s = 0., 0., 0.
        val_hr_s_list, val_ndcg_s_list, val_mrr_s_list = [], [], []
        val_hr_t_list, val_ndcg_t_list, val_mrr_t_list = [], [], []
        # loss computation list

        loss_d_s_list = []
        loss_g_s_list = []
        loss_t_list = []

        t_start = time.time()
        t_source_total= 0.
        t_source_each_epoch =0.
        e_source_total= 0
        for e in range(n_epochs):
            if (e % self.t_s_epochs) < self.s_epochs:
                t_source_start= time.time()
            # if (e< 30 and self.dataset_name.find("amazon")!=-1) or (e< 20 and self.dataset_name.find("amazon")==-1):
            # if e < 30:
                # train model s
                epoch_d_loss, epoch_g_loss, epoch_time_s = self.train_model_s(e)
                loss_d_s_list.append(epoch_d_loss)
                loss_g_s_list.append(epoch_g_loss)
                t_source_each_epoch = time.time() -t_source_start
                e_source_total+=1
                t_source_total+=t_source_each_epoch
                print(e_source_total, t_source_total/e_source_total,'---------------------')


                if do_eval_model_s and (e % n_epochs_per_eval == 0):
                    self.model_s.eval()
                    self.model_t.eval()
                    avg_hr_s, avg_ndcg_s, avg_mrr_s = test_process(self.model_s,
                                                                   self.train_loader,
                                                                   self.feed_data,
                                                                   self.device,
                                                                   topK=topK_list[1],
                                                                   dr_target="sdr",
                                                                   mode="val",
                                                                   domain="s")
                    # avg_hr_s, avg_ndcg_s, avg_mrr_s = test_process({'user_embed': self.user_embeddings,
                    #                                                 'gen_s': self.model_s},
                    #                                                self.train_loader,
                    #                                                self.feed_data,
                    #                                                self.device,
                    #                                                topK=topK_list[1],
                    #                                                dr_target="sdr", mode="val",
                    #                                                domain="s")
                    wandb.log({
                        'epoch': e,
                        'val_hr_s': avg_hr_s,
                        'val_ndcg_s': avg_ndcg_s,
                        'val_mrr_s': avg_mrr_s,

                    })
                    val_hr_s_list.append(avg_hr_s)
                    val_ndcg_s_list.append(avg_ndcg_s)
                    val_mrr_s_list.append(avg_mrr_s)

                    val_log_str = f'val in fitting: source: hr: {avg_hr_s}, ndcg: {avg_ndcg_s}, mrr: {avg_mrr_s}'
                    print(val_log_str)
                    with (self.logPath / 'tmp.txt').open('a') as fw:
                        fw.write(val_log_str)

                    if avg_hr_s > best_hr_s:
                        torch.save(self.model_s.state_dict(), (self.logPath / f'best_hr_s.pkl'))
                        best_hr_s = avg_hr_s
                    if avg_ndcg_s > best_ndcg_s:
                        torch.save(self.model_s.state_dict(), (self.logPath / f'best_ndcg_s.pkl'))
                        best_ndcg_s = avg_ndcg_s
                    if avg_mrr_s > best_mrr_s:
                        torch.save(self.model_s.state_dict(), (self.logPath / f'best_mrr_s.pkl'))
                        best_mrr_s = avg_mrr_s

            else:
                # train model t
                epoch_loss_t, epoch_time_t = self.train_model_t(e)
                loss_t_list.append(epoch_loss_t)

                if do_eval_model_t and (e % n_epochs_per_eval == 0):
                    # avg_hr_t, avg_ndcg_t, avg_mrr_t = test_process({'gen_s': self.model_s,
                    #                                                 'model_t': self.model_t},
                    #                                                self.train_loader,
                    #                                                self.feed_data,
                    #                                                self.device,
                    #                                                topK=topK_list[1],
                    #                                                dr_target="cdr")
                    self.model_s.eval()
                    self.model_t.eval()
                    avg_hr_t, avg_ndcg_t, avg_mrr_t = test_process(self.model_t,
                                                                   self.train_loader,
                                                                   self.feed_data, self.device,
                                                                   topK_list[1], dr_target="cdr",
                                                                   mode='val')

                    wandb.log({
                        'epoch': e,
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
                        torch.save(self.model_t.state_dict(), (self.logPath / 'best_hr_t.pkl'))
                        best_hr_t = avg_hr_t
                    if avg_ndcg_t > best_ndcg_t:
                        torch.save(self.model_t.state_dict(), (self.logPath / 'best_ndcg_t.pkl'))
                        best_ndcg_t = avg_ndcg_t
                    if avg_mrr_t > best_mrr_t:
                        torch.save(self.model_t.state_dict(), (self.logPath / 'best_mrr_t.pkl'))
                        best_mrr_t = avg_mrr_t
                self.scheduler_t.step()
        t_total= time.time()- t_start
        # t_source_each_epoch = time.time() - t_source_each_epoch
        # e_source_total += 1
        # t_source_total += t_source_each_epoch
        t_avg_source= t_source_total/e_source_total


        print(f", t_total: {t_total}, t_source_total: {t_source_total}, t_source_avg: {t_avg_source}, one_last_source_epoch: {t_source_each_epoch}")
        '''
            model_R =self.model_s.generator_s(torch.from_numpy(np.array(range(self.num_users))).to('cuda'))[0]
            import pickle
            pickle.dump(model_R.detach().cpu(), open("fake_ratings_50.pkl", 'wb'))
        '''
        loss_model_s_dict = {
            'd_loss': loss_d_s_list,
            'g_loss': loss_g_s_list,
        }
        plot_loss(loss_model_s_dict, self.logPath, loss_fig_title="cdr_gan_gnn_agent_loss_model_s")

        loss_model_t_dict = {
            't_loss': loss_t_list,
        }
        plot_loss(loss_model_t_dict, self.logPath, loss_fig_title="cdr_gan_gnn_agent_loss_model_t")

    def testing(self, topK_list=[], do_test_s=False, do_test_t=False):
        if do_test_s:
            self.testing_s(topK_list)

        if do_test_t:
            self.testing_t(topK_list)

    def testing_s(self, topK_list=[]):
        self.model_s.eval()
        print("Testing source model ...")
        for topK in topK_list:
            self.model_s.load_state_dict(torch.load((self.logPath / f'best_hr_s.pkl')))
            test_hr_s, _, _ = test_process(self.model_s, self.train_loader, self.feed_data, self.device,
                                           topK, dr_target="sdr", mode="test", domain="s")
            # test_hr_t, _, _ = test_process({'user_embed': self.user_embeddings,
            #                                 'gen_s': self.model_s,
            #                                 }, self.train_loader, self.feed_data, self.device,
            #                                topK, dr_target="sdr", mode="test", domain="s")

            self.model_s.load_state_dict(torch.load((self.logPath / f'best_ndcg_s.pkl')))
            _, test_ndcg_s, _ = test_process(self.model_s, self.train_loader, self.feed_data, self.device,
                                             topK, dr_target="sdr", mode="test", domain="s")
            # _, test_ndcg_t, _ = test_process({'user_embed': self.user_embeddings,
            #                                   'gen_s': self.model_s,
            #                                   }, self.train_loader, self.feed_data, self.device,
            #                                  topK, dr_target="sdr", mode="test", domain="s")
            self.model_s.load_state_dict(torch.load((self.logPath / f'best_mrr_s.pkl')))
            _, _, test_mrr_s = test_process(self.model_s, self.train_loader, self.feed_data, self.device,
                                            topK, dr_target="sdr", mode="test", domain="s")
            # _, _, test_mrr_t = test_process({'user_embed': self.user_embeddings,
            #                                  'gen_s': self.model_s,
            #                                                 }, self.train_loader, self.feed_data, self.device,
            #                                 topK, dr_target="sdr", mode="test", domain="s")

            wandb.log({
                f'test_hr_s_{topK}': test_hr_s,
                f'test_ndcg_s_{topK}': test_ndcg_s,
                f'test_mrr_s_{topK}': test_mrr_s,
            })

            test_los_str = f"Test model s: TopK:{topK} ----> Source: hr: {test_hr_s}, ndcg: {test_ndcg_s} mrr: {test_mrr_s}"
            print(test_los_str)
            with (self.logPath / 'tmp.txt').open('a') as fw:
                fw.write(test_los_str)

    def testing_t(self, topK_list=[]):
        self.model_t.eval()
        self.model_s.eval()

        print("Testing target model ...")
        model_best_hr = torch.load((self.logPath / 'best_hr_t.pkl'))
        model_best_ndcg = torch.load((self.logPath / 'best_ndcg_t.pkl'))
        model_best_mrr = torch.load((self.logPath / 'best_mrr_t.pkl'))

        for topK in topK_list:
            self.model_t.load_state_dict(model_best_hr)
            test_hr_t, _, _ = test_process(self.model_t, self.train_loader, self.feed_data, self.device,
                                           topK, dr_target='cdr', mode="test")
            self.model_t.load_state_dict(model_best_ndcg)
            _, test_ndcg_t, _ = test_process(self.model_t, self.train_loader, self.feed_data, self.device,
                                             topK, dr_target='cdr', mode="test")
            self.model_t.load_state_dict(model_best_mrr)
            _, _, test_mrr_t = test_process(self.model_t, self.train_loader, self.feed_data, self.device,
                                            topK, dr_target='cdr', mode="test")

            # self.model_t.load_state_dict(model_best_hr)
            # test_hr_t, _, _ = test_process({'gen_s': self.model_s,
            #                                 'model_t': self.model_t},
            #                                self.train_loader,
            #                                self.feed_data,
            #                                self.device,
            #                                topK=topK,
            #                                dr_target="cdr", mode="test")
            #
            # self.model_t.load_state_dict(model_best_ndcg)
            # _, test_ndcg_t, _ = test_process({'gen_s': self.model_s,
            #                                   'model_t': self.model_t},
            #                                  self.train_loader,
            #                                  self.feed_data,
            #                                  self.device,
            #                                  topK=topK,
            #                                  dr_target="cdr", mode="test")
            # self.model_t.load_state_dict(model_best_mrr)
            # _, _, test_mrr_t = test_process({'gen_s': self.model_s,
            #                                  'model_t': self.model_t},
            #                                 self.train_loader,
            #                                 self.feed_data,
            #                                 self.device,
            #                                 topK=topK,
            #                                 dr_target="cdr", mode="test")
            wandb.log({
                f'test_hr_t_{topK}': test_hr_t,
                f'test_ndcg_t_{topK}': test_ndcg_t,
                f'test_mrr_t_{topK}': test_mrr_t,
            })
            test_los_str = f"Test model t TopK:{topK} --> target: hr: {test_hr_t}, ndcg: {test_ndcg_t} mrr: {test_mrr_t}"
            print(test_los_str)
            with (self.logPath / 'tmp.txt').open('a') as fw:
                fw.write(test_los_str)


def main_sweep():
    with wandb.init(magic=True, name=f'PPGenCDR'):
        with Path(wandb.config.model_config).open("r") as f:
            config_proj = yaml.safe_load(f)
            wandb.config.update(config_proj)
            args = argparse.Namespace(**wandb.config)
            print('running.....')
            print(args)
            finetune_goal=  {"1":"change autoencoder+ change epoch condition",
                            }
            wandb.log({
                'goal': finetune_goal["1"] # ["3"]
            })

            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            args.device = DEVICE
            agent = CDRGanGNNAgent(args)
            agent.fit(args.epochs, 1, [5, 10], use_pretrain=False)
            agent.testing([5, 10], do_test_s=False, do_test_t=True)
            print(args)


def main(model_config, mode):
    with wandb.init(magic=True, name=f'PPGenCDR', mode=mode):
        with Path(model_config).open("r") as f:
            config_proj = yaml.safe_load(f)
            wandb.config.update(config_proj)
            args = argparse.Namespace(**wandb.config)
            print('running.....')
            print(args)
            finetune_goal=  {"1":"change autoencoder+ change epoch condition",
                            }
            wandb.log({
                'goal': finetune_goal["1"] # ["3"]
            })

            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            args.device = DEVICE
            agent = CDRGanGNNAgent(args)
            agent.fit(args.epochs, 1, [5, 10], use_pretrain=False)
            agent.testing([5, 10], do_test_s=False, do_test_t=True)
            print(args)

if __name__ == '__main__':

    model_configs = [
        '../config/cdr_gan_dmf_param_ssl_al_pid_amazon_m2b.yaml',
        # # #
        #   '../config/cdr_gan_dmf_param_ssl_al_pid_amazon2_m2mu.yaml',
        #   '../config/cdr_gan_dmf_param_ssl_al_pid_amazon3_mu2b.yaml',
        # '../config/cdr_gan_dmf_param_ssl_al_pid_douban_b2mu.yaml',
        # '../config/cdr_gan_dmf_param_ssl_al_pid_douban_m2b.yaml',
        # '../config/cdr_gan_dmf_param_ssl_al_pid_douban_m2mu.yaml'
    ]
    # # note use sweep ##############################################
    # for conf in model_configs:
    #     sweep_config = yaml.safe_load(Path("../config/cdr_gan_dmf_param_ssl_al_pid_douban_sweep.yml").open("r"))
    #     sweep_config["parameters"]["model_config"]["value"] = conf
    #     sweep_id = wandb.sweep(sweep_config, project="ETL")
    #     wandb.agent(sweep_id, function=main_sweep, count=3)

    # note ######################  not wandb sweep version #########
    for conf in model_configs:
        main(conf, mode='disabled')
