from pathlib import Path
import sys
DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME.parent))
sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent))
import numpy as np
from scipy.sparse import csr_matrix
from torch import nn, FloatTensor
import torch
from utils.model_utils import load_model
from utils.param_utils import Summary
from evaluation import test_process
import random
import uuid
from data.dataset import Dataset, DatasetDouban, DatasetAmazon
# from dataset import Dataset, DatasetDouban, DatasetAmazon
import json


class RecAgent():
    def __init__(self, args):
        uuid_number = uuid.uuid1() if args.uuid_tag is None else args.uuid_tag
        log = Path(args.log_prefix).joinpath(args.method).joinpath(
            f'{args.dataset}_{args.source}_{args.t_percent}_{args.target}_{args.s_percent}_{uuid_number}')

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        self.logPath = log
        if not self.logPath.exists():
            self.logPath.mkdir(parents=True, exist_ok=True)

        with (log / f"tmp.txt").open('a') as fw:
            fw.write(json.dumps(vars(args)))

        if args.DataIn == 'amazon':
            dataset = DatasetAmazon
        elif args.DataIn == "douban":
            dataset = DatasetDouban  # careful 这里没有加上对应的dataset
        else:
            dataset = Dataset

        self.dataset = dataset(args.batch_size, dataset=args.dataset, processed_data_dir=args.processed_data_dir,
                               use_discrete=args.use_discrete, re_generate=args.re_generate)
        self.num_users = self.dataset.num_user
        self.num_items_s = self.dataset.num_source
        self.num_items_t = self.dataset.num_target
        self.batch_size = args.batch_size
        self.device = args.device

        # prepare for rating embeddings
        print("preparing the training data ...")
        row, col = self.dataset.get_part_train_indices('source', args.s_percent)
        values = np.ones(row.shape[0])
        user_s = csr_matrix((values, (row, col)), shape=(self.num_users, self.num_items_s)).toarray()
        self.user_ratings_s = FloatTensor(user_s).to(self.device)  # (self.user_ratings_s==2).nonzero()
        self.user_ratings_s[(self.user_ratings_s == 2)] = 1

        row, col = self.dataset.get_part_train_indices('target', args.t_percent)
        values = np.ones(row.shape[0])
        user_t = csr_matrix((values, (row, col)), shape=(self.num_users, self.num_items_t)).toarray()
        self.user_ratings_t = FloatTensor(user_t).to(self.device)
        self.user_ratings_t[(self.user_ratings_t == 2)] = 1

        print("Preparing the training data over ...")

        # for shared user one-hot representation
        self.user_ids = np.arange(self.num_users).reshape([self.num_users, 1])
        self.pretrain_user_ids = self.user_ids[: int(args.p_percent * self.num_users)]
        self.train_user_ids = self.user_ids[int(args.p_percent * self.num_users):]
        if len(self.pretrain_user_ids) > 0:
            self.pretrain_loader = torch.utils.data.DataLoader(torch.from_numpy(self.pretrain_user_ids),
                                                               batch_size=args.batch_size,
                                                               shuffle=True)
        else:
            self.pretrain_loader = None
        self.train_loader = torch.utils.data.DataLoader(torch.from_numpy(self.train_user_ids),
                                                        batch_size=args.batch_size,
                                                        shuffle=True)
        # prepare data fot test process
        source_val, source_test, source_neg = self.dataset.source_val, self.dataset.source_test, self.dataset.source_neg
        target_val, target_test, target_neg = self.dataset.target_val, self.dataset.target_test, self.dataset.target_neg
        # feed data for testing
        self.feed_data = {
            "fts_s": user_s,
            "fts_t": user_t,
            "source_val": source_val,
            "source_test": source_test,
            "source_neg": source_neg,
            "target_val": target_val,
            "target_test": target_test,
            "target_neg": target_neg,
        }

    def pretrain(self, n_epochs, topK_list=None):
        pass

    def testing(self):
        raise NotImplementedError("Testing must be implemented")

    def fit(self, n_epochs, n_epochs_per_eval, topK_list=[]):
        # def fit(self, n_epochs, n_epochs_per_eval, topK_list=[], use_pretrain=True, model_path=None,
        #         use_model_tag='best_hr_s_pretrain', strict=False):
        raise NotImplementedError("Fit must be implemented")

    def train_one_epoch(self, epoch):
        raise NotImplementedError("Train_one_epoch must be implemented")


class RecAgentchgLoader(RecAgent):
    def __init__(self, args):
        super(RecAgentchgLoader, self).__init__(args)

    def change_loader_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size
        if len(self.pretrain_user_ids) > 0:
            self.pretrain_loader = torch.utils.data.DataLoader(torch.from_numpy(self.pretrain_user_ids),
                                                               batch_size=new_batch_size,
                                                               shuffle=True)
        else:
            self.pretrain_loader = None
        self.train_loader = torch.utils.data.DataLoader(torch.from_numpy(self.train_user_ids),
                                                        batch_size=new_batch_size,
                                                        shuffle=True)


class RecAgentNegSample(RecAgent):
    def __init__(self, args):
        super(RecAgentNegSample, self).__init__(args)

    def negative_samples(self, domain='s', K=1, return_dict=False, use_diff_neg_per_pos=False):
        S = []
        S_dict = {}
        for i, user in enumerate(self.user_ids):
            user = user[0]
            posForUser = self.dataset.user2item_source[user] if domain.lower() == 's' else \
                self.dataset.user2item_target[user]
            if len(posForUser) == 0:
                print(posForUser, user)
                raise RuntimeError('user has no samples')
            if use_diff_neg_per_pos:  # 返回pos 的条件就是对应负采样的结果不一样,否则这样不成立
                ns_list_per_user= []

                for pos_i in posForUser:
                    negItemList = []
                    while len(negItemList) < K:
                        while True:
                            negitem = np.random.randint(0,
                                                        self.num_items_s if domain.lower() == 's' else self.num_items_t)
                            if negitem in posForUser:
                                continue
                            else:
                                break
                        negItemList.append(negitem)
                    ns_list_per_user.append(negItemList)  # 每个pos item 对应不同的negative items

                if not return_dict:
                    S.append([user,posForUser, ns_list_per_user])
                else:
                    S_dict[str(user)]= (posForUser, ns_list_per_user)
            else:  # 每个user 单独负采样在每个epoch 重新采样的情况下是不受影响的
                # shuffling pos
                posIndex = np.random.randint(0, len(posForUser))
                posItem = posForUser[posIndex]
                negItemList = []
                while len(negItemList) < K:
                    while True:
                        negitem = np.random.randint(0, self.num_items_s if domain.lower() == 's' else self.num_items_t)
                        if negitem in posForUser:
                            continue
                        else:
                            break
                    negItemList.append(negitem)
                if not return_dict:
                    S.append([user, posItem, negItemList])
                else:
                    S_dict[str(user)] = (posItem,
                                         negItemList)  # shuffling positive items w/ similar neg  一个采样的positive item 以及多个negative itesm

        if return_dict:
            return S_dict
        else:
            return S


class RecAgentCDRCL(RecAgentchgLoader, RecAgentNegSample):
    def __init__(self, args):
        super(RecAgentCDRCL, self).__init__(args)
        # note 亲测可用


# if __name__ == '__main__':
#     from config_argparser import ConfigArgumentParser, YamlConfigAction
#
#
#     parser = ConfigArgumentParser('test')
#     parser.add_argument("--config", action=YamlConfigAction, default=['../config/cdr_gan_gnn_param_ssl.yaml'])
#     args = parser.parse_args()
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     args.device = DEVICE
#     ra = RecAgentCDRCL(args)
#     ra.negative_samples()
#     ra.change_loader_batch_size(2)
#     print(ra.batch_size)
