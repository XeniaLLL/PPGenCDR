import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn.utils import shuffle
from pathlib import Path
import sys
import json
# from sklearn.cluster import KMeans
from torch.nn import functional
from copy import deepcopy

sys.path.append("..")


class Dataset(object):
    def __init__(self, BATCH_SIZE, dataset='amazon', source="movie", target='book',
                 processed_data_dir='processed_data', re_generate=False, **kwargs):
        self.batch_size = BATCH_SIZE
        self.source = source
        self.target = target
        self.processed_data_dir = Path(f'{processed_data_dir}/{dataset}')
        self.re_generate = re_generate
        # self.processed_data_dir = Path(f'processed_data/{dataset}')
        self.user2item_source, self.user2item_target, self.item2user_source, self.item2user_target = self.load_data(
            dataset)
        '''
        user2item_source: dict {user: item} in source domain
        '''
        self.source_set = self.item2user_source.keys()
        self.target_set = self.item2user_target.keys()
        self.num_user = len(self.user2item_source)
        self.num_source = len(self.item2user_source)
        self.num_target = len(self.item2user_target)

        print(f"Finish loading: {self.num_user} users, {self.num_source} source data, {self.num_target} target data")
        print('gen testing dataset ...')
        self.init_test_data()
        print("Finish loading ...")
        self.test_count = 0
        self.batch_count = 0
        self.count = 0
        self.epoch = 1

    def init_test_data(self):
        self.source_val, self.source_test, self.source_neg, self.target_val, self.target_test, self.target_neg = self.get_test_data()

    def load_data(self, dataset):
        print(f"Loading data: {dataset}")
        user2_item_source = pickle.load((self.processed_data_dir / f'peo2{self.source}_id.pkl').open('rb'))
        user2_item_target = pickle.load((self.processed_data_dir / f'peo2{self.target}_id.pkl').open('rb'))
        source_item_2user = pickle.load((self.processed_data_dir / f'{self.source}2peo_id.pkl').open('rb'))
        target_item_2user = pickle.load((self.processed_data_dir / f'{self.target}2peo_id.pkl').open('rb'))

        return user2_item_source, user2_item_target, source_item_2user, target_item_2user

    def get_part_train_indices(self, domain, precent):
        row, col = [], []

        if domain == 'source':
            dict = self.user2item_source
            val = self.source_val
            test = self.source_test
        elif domain == 'target':
            dict = self.user2item_target
            val = self.target_val
            test = self.target_test
        else:
            return

        for user in dict:

            if len(dict[user]) > 2:
                dict[user].remove(val[user])
                dict[user].remove(test[user])
            else:
                dict[user].remove(val[user])
            dict[user] = shuffle(dict[user], random_state=72)
            num_item = int(round(precent * len(dict[user])))
            for i in range(num_item):  # 统一每个user对应的样本量
                row.append(user)
                col.append(dict[user][i])

        row = np.array(row)
        col = np.array(col)

        return row, col

    def gen_sample_eval_set(self, domain):
        '''

        :param domain:
        :return:
        '''
        val = {}
        test = {}
        neg = {}

        for i in tqdm(self.user2item_source if domain == "s" else self.user2item_target):
            items = self.user2item_source[i] if domain == "s" else self.user2item_target[i]  # 迭代user 得到items
            if len(items) >= 2:
                items_choices = shuffle(items, random_state=2020)[:2]
                val[i] = items_choices[0]
                test[i] = items_choices[1]
            else:  # 只有一个就同时用做这个 user 的val & test
                val[i] = items[0]
                test[i] = items[0]

            all_neg_source = list(
                (set(list(range(self.num_source if domain == "s" else self.num_target))) - set(items)))
            neg[i] = shuffle(all_neg_source, random_state=2020)[:99]  # 所有没有交互的选前99个负样本

        return val, test, neg

    def get_test_data(self):
        if (not (self.processed_data_dir / f'{self.source}_val.pkl').is_file() or \
            not (self.processed_data_dir / f'{self.source}_val.pkl').is_file()) or self.re_generate:
            source_val, source_test, source_neg = self.gen_sample_eval_set('s')
            target_val, target_test, target_neg = self.gen_sample_eval_set('t')

            pickle.dump(source_val, (self.processed_data_dir / f'{self.source}_val.pkl').open('wb'))
            pickle.dump(target_val, (self.processed_data_dir / f'{self.target}_val.pkl').open('wb'))

            pickle.dump(source_test, (self.processed_data_dir / f'{self.source}_test.pkl').open('wb'))
            pickle.dump(target_test, (self.processed_data_dir / f'{self.target}_test.pkl').open('wb'))

            pickle.dump(source_neg, (self.processed_data_dir / f'{self.source}_neg.pkl').open('wb'))
            pickle.dump(target_neg, (self.processed_data_dir / f'{self.target}_neg.pkl').open('wb'))

        else:
            source_val = pickle.load((self.processed_data_dir / f'{self.source}_val.pkl').open('rb'))
            target_val = pickle.load((self.processed_data_dir / f'{self.target}_val.pkl').open('rb'))

            source_test = pickle.load((self.processed_data_dir / f'{self.source}_test.pkl').open('rb'))
            target_test = pickle.load((self.processed_data_dir / f'{self.target}_test.pkl').open('rb'))

            source_neg = pickle.load((self.processed_data_dir / f'{self.source}_neg.pkl').open('rb'))
            target_neg = pickle.load((self.processed_data_dir / f'{self.target}_neg.pkl').open('rb'))
        return source_val, source_test, source_neg, target_val, target_test, target_neg


from transformers import AutoModel, AutoTokenizer
import torch


class DatasetSideInfo(Dataset):
    def __init__(self, BATCH_SIZE, dataset='amazon', source="movie", target='book',
                 processed_data_dir='processed_data', device="cuda", use_discrete=False, re_generate=False):
        self.device = device
        self.use_discrete = use_discrete
        self.re_generate = re_generate
        super(DatasetSideInfo, self).__init__(BATCH_SIZE=BATCH_SIZE, dataset=dataset, source=source, target=target,
                                              processed_data_dir=processed_data_dir)

        # print("Finish init")
        # for param in self.model.parameters(): # note 等价于在后面加with torch.no_grad()
        #     param.requires_grad = False
        # self.model.eval()

    def init_test_data(self):
        self.source_val, self.source_test, self.source_neg, self.source_user_side_info, self.source_total_item_id, self.target_val, self.target_test, self.target_neg, self.target_user_side_info, self.target_total_item_id = self.get_test_data(
            use_discrete=True, re_generate=True)

    def get_part_train_indices(self, domain, precent):
        row, col = [], []

        if domain == 'source':
            dict = self.source_total_item_id
            val = self.source_val
            test = self.source_test
        elif domain == 'target':
            dict = self.target_total_item_id
            val = self.target_val
            test = self.target_test
        else:
            return

        for user in dict:
            if len(dict[user]) > 2:
                dict[user].remove(val[user])
                dict[user].remove(test[user])
            else:
                dict[user].remove(val[user])
            user_item_set = list(set(dict[user]))
            user_item_set = shuffle(user_item_set, random_state=72)
            num_item = int(round(precent * len(user_item_set)))
            for i in range(num_item):  # 统一每个user对应的样本量
                row.append(user)
                col.append(user_item_set[i])
                # if user_item_set[i]==1281 or user_item_set[i]==1280:
                #     print("Notice----------------------",dict[user], user)

        row = np.array(row)
        col = np.array(col)

        return row, col

    def load_data(self, dataset):
        print(f"Loading data: {dataset}")
        user2_item_source = pickle.load((self.processed_data_dir / f'peo2{self.source}_id_si.pkl').open('rb'))
        user2_item_target = pickle.load((self.processed_data_dir / f'peo2{self.target}_id_si.pkl').open('rb'))
        source_item_2user = pickle.load((self.processed_data_dir / f'{self.source}2peo_id_si.pkl').open('rb'))
        target_item_2user = pickle.load((self.processed_data_dir / f'{self.target}2peo_id_si.pkl').open('rb'))

        return user2_item_source, user2_item_target, source_item_2user, target_item_2user

    def gen_sample_eval_set(self, domain):
        '''

        :param domain:
        :return:
        '''
        u_items = {}
        user_side_info = {}
        val_item_id = {}
        test_item_id = {}
        neg_item_id = {}

        for i in tqdm(self.user2item_source if domain == "s" else self.user2item_target):
            items = self.user2item_source[i] if domain == "s" else self.user2item_target[i]  # 迭代user 得到items
            items_id = [k[0] for k in items]
            u_items[i] = items_id

            # items_reviews = [k[1] for k in items]
            # # todo IMPLEMENT adding user profile info
            # # side_info_embed = self.get_emb(items_reviews)
            # user_side_info[i] = items_reviews.mean(-1)

            if len(items) >= 2:
                items_choices = shuffle(items, random_state=2020)[:2]
                val_item_id[i] = items_choices[0][0]
                test_item_id[i] = items_choices[1][0]

            else:  # 只有一个就同时用做这个 user 的val & test
                val_item_id[i] = items[0][0]
                test_item_id[i] = items[0][0]

            all_neg_source = list(
                (set(list(range(self.num_source if domain == "s" else self.num_target))) - set(items_id)))
            neg_item_id[i] = shuffle(all_neg_source, random_state=2020)[:99]  # 所有没有交互的选前99个负样本

        return val_item_id, test_item_id, neg_item_id, user_side_info, u_items

    def discrete_si(self, u_si):

        user_side_info_embs = list(u_si.values())
        # lrmodel = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=None,
        #                  copy_x=True, algorithm='auto')

        cluster_ids_x, cluster_centers = kmeans(
            X=torch.stack(user_side_info_embs), num_clusters=2, distance='euclidean', device=self.device
        )
        labels_one_hot = [functional.one_hot(cluster_ids_x[i], 2) for i in range(len(cluster_ids_x))]
        user_side_info_discrete = {}
        for idx, u_id in enumerate(u_si):
            user_side_info_discrete[u_id] = labels_one_hot[idx]

        return user_side_info_discrete

    def get_test_data(self):
        # source_val, source_test, source_neg, source_user_side_info, source_total_item_id = self.gen_sample_eval_set(
        #     's')
        # target_val, target_test, target_neg, target_user_side_info, target_total_item_id = self.gen_sample_eval_set(
        #     't')
        if (not (self.processed_data_dir / f'{self.source}_val.pkl').is_file() or \
            not (self.processed_data_dir / f'{self.source}_val.pkl').is_file()) or self.re_generate:
            source_val, source_test, source_neg, source_user_side_info, source_total_item_id = self.gen_sample_eval_set(
                's')
            target_val, target_test, target_neg, target_user_side_info, target_total_item_id = self.gen_sample_eval_set(
                't')

            if self.use_discrete:
                source_user_side_info = self.discrete_si(source_user_side_info)
                target_user_side_info = self.discrete_si(target_user_side_info)

            pickle.dump(source_total_item_id, (self.processed_data_dir / f'{self.source}_total_items.pkl').open('wb'))
            pickle.dump(target_total_item_id, (self.processed_data_dir / f'{self.target}_total_items.pkl').open('wb'))

            pickle.dump(source_val, (self.processed_data_dir / f'{self.source}_val.pkl').open('wb'))
            pickle.dump(target_val, (self.processed_data_dir / f'{self.target}_val.pkl').open('wb'))

            pickle.dump(source_test, (self.processed_data_dir / f'{self.source}_test.pkl').open('wb'))
            pickle.dump(target_test, (self.processed_data_dir / f'{self.target}_test.pkl').open('wb'))

            pickle.dump(source_neg, (self.processed_data_dir / f'{self.source}_neg.pkl').open('wb'))
            pickle.dump(target_neg, (self.processed_data_dir / f'{self.target}_neg.pkl').open('wb'))

            pickle.dump(source_user_side_info,
                        (self.processed_data_dir / f'{self.source}_si_use_discrete_{self.use_discrete}.pkl').open('wb'))
            pickle.dump(target_user_side_info,
                        (self.processed_data_dir / f'{self.target}_si_use_discrete_{self.use_discrete}.pkl').open('wb'))

        else:

            source_total_item_id = pickle.load((self.processed_data_dir / f'{self.source}_total_items.pkl').open('rb'))
            target_total_item_id = pickle.load((self.processed_data_dir / f'{self.target}_total_items.pkl').open('rb'))

            source_val = pickle.load((self.processed_data_dir / f'{self.source}_val.pkl').open('rb'))
            target_val = pickle.load((self.processed_data_dir / f'{self.target}_val.pkl').open('rb'))

            source_test = pickle.load((self.processed_data_dir / f'{self.source}_test.pkl').open('rb'))
            target_test = pickle.load((self.processed_data_dir / f'{self.target}_test.pkl').open('rb'))

            source_neg = pickle.load((self.processed_data_dir / f'{self.source}_neg.pkl').open('rb'))
            target_neg = pickle.load((self.processed_data_dir / f'{self.target}_neg.pkl').open('rb'))

            source_user_side_info = pickle.load(
                (self.processed_data_dir / f'{self.source}_si_use_discrete_{self.use_discrete}.pkl').open('rb'))
            target_user_side_info = pickle.load(
                (self.processed_data_dir / f'{self.target}_si_use_discrete_{self.use_discrete}.pkl').open('rb'))
        return source_val, source_test, source_neg, source_user_side_info, source_total_item_id, target_val, target_test, target_neg, target_user_side_info, target_total_item_id


class DatasetAmazonDiscrete(DatasetSideInfo):
    def __init__(self, BATCH_SIZE, dataset='amazon', source="movie", target='book',
                 processed_data_dir='processed_data', device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("activebus/BERT-XD_Review")
        self.model = AutoModel.from_pretrained("activebus/BERT-XD_Review")
        self.model.to(device)
        super(DatasetAmazonDiscrete, self).__init__(BATCH_SIZE=BATCH_SIZE, dataset=dataset, source=source,
                                                    target=target,
                                                    processed_data_dir=processed_data_dir, device=device)

    def get_emb(self, text):
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')  # 为了维护数据的结构,一条条输入算了
        tokens_tensor = tokens['input_ids'].to(self.device)  # .unsqueeze(0)
        segments_tensors = tokens['token_type_ids'].to(self.device)  # .unsqueeze(0)
        attention_mask_tensors = tokens['attention_mask'].to(self.device)  # .unsqueeze(0)

        with torch.no_grad():
            summary_embeddings = self.model(tokens_tensor, segments_tensors, attention_mask_tensors)[0][:, 0,
                                 :].mean(0)
        return summary_embeddings


class DatasetAmazon(DatasetSideInfo):
    def __init__(self, BATCH_SIZE, dataset='amazon', source="movie", target='book',
                 processed_data_dir='processed_data', device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("activebus/BERT-XD_Review")
        self.model = AutoModel.from_pretrained("activebus/BERT-XD_Review")
        self.model.to(device)
        super(DatasetAmazon, self).__init__(BATCH_SIZE=BATCH_SIZE, dataset=dataset, source=source, target=target,
                                            processed_data_dir=processed_data_dir, device=device)

    def get_emb(self, text):
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')  # 为了维护数据的结构,一条条输入算了
        tokens_tensor = tokens['input_ids'].to(self.device)  # .unsqueeze(0)
        segments_tensors = tokens['token_type_ids'].to(self.device)  # .unsqueeze(0)
        attention_mask_tensors = tokens['attention_mask'].to(self.device)  # .unsqueeze(0)

        with torch.no_grad():
            summary_embeddings = self.model(tokens_tensor, segments_tensors, attention_mask_tensors)[0][:, 0,
                                 :].mean(0)
        return summary_embeddings

    def gen_sample_eval_set(self, domain):
        '''

        :param domain:
        :return:
        '''
        u_items = {}
        user_side_info = {}
        val_item_id = {}
        test_item_id = {}
        neg_item_id = {}

        for i in tqdm(self.user2item_source if domain == "s" else self.user2item_target):
            items = self.user2item_source[i] if domain == "s" else self.user2item_target[i]  # 迭代user 得到items
            items_id = [k[0] for k in items]
            u_items[i] = items_id

            items_reviews = [k[1] for k in items]
            side_info_embed = self.get_emb(items_reviews)
            user_side_info[i] = side_info_embed

            if len(items) >= 2:
                items_choices = shuffle(items, random_state=2020)[:2]
                val_item_id[i] = items_choices[0][0]
                test_item_id[i] = items_choices[1][0]

            else:  # 只有一个就同时用做这个 user 的val & test
                val_item_id[i] = items[0][0]
                test_item_id[i] = items[0][0]

            all_neg_source = list(
                (set(list(range(self.num_source if domain == "s" else self.num_target))) - set(items_id)))
            neg_item_id[i] = shuffle(all_neg_source, random_state=2020)[:99]  # 所有没有交互的选前99个负样本

        return val_item_id, test_item_id, neg_item_id, user_side_info, u_items

    def discrete_si(self, u_si):

        user_side_info_embs = list(u_si.values())
        # lrmodel = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=None,
        #                  copy_x=True, algorithm='auto')

        cluster_ids_x, cluster_centers = kmeans(
            X=torch.stack(user_side_info_embs), num_clusters=2, distance='euclidean', device=self.device
        )
        labels_one_hot = [functional.one_hot(cluster_ids_x[i], 2) for i in range(len(cluster_ids_x))]
        user_side_info_discrete = {}
        for idx, u_id in enumerate(u_si):
            user_side_info_discrete[u_id] = labels_one_hot[idx]

        return user_side_info_discrete


# class DatasetAmazon2(DatasetSideInfo):
#     def __init__(self, BATCH_SIZE, dataset='amazon', source="movie", target='book',
#                  processed_data_dir='processed_data', device="cuda"):
#         super(DatasetAmazon2, self).__init__(BATCH_SIZE=BATCH_SIZE, dataset=dataset, source=source, target=target,
#                                             processed_data_dir=processed_data_dir, device=device)

dataset_name_maping_dict = {
    "movie": "movie",
    "music": "cd",
    "book": "book",
    "video": "video",
    "clothes": "clothes"
}


class DatasetAmazon2(object):
    def __init__(self, BATCH_SIZE, dataset='amazon', source="movie", target='book',
                 processed_data_dir='processed_data', device="cuda"):

        self.train_path_t = Path('amazon_model/' + dataset_name_maping_dict[target] + '/train_data.json')
        self.train_path_s = Path('amazon_model/' + dataset_name_maping_dict[source] + '/train_data.json')
        self.pretrain_path_t = Path('amazon_model/' + dataset_name_maping_dict[target] + '/latent_embeddings.json')
        self.pretrain_path_s = Path(
            'amazon_model/' + dataset_name_maping_dict[source] + '/latent_embeddings.json')  # note U_s,　V_s
        self.review_path_t = Path('amazon_model/' + dataset_name_maping_dict[target] + '/review_embeddings.json')
        self.review_path_s = Path('amazon_model/' + dataset_name_maping_dict[source] + '/review_embeddings.json')

        self.batch_size = BATCH_SIZE
        self.processed_data_dir = Path(f'{processed_data_dir}/{dataset}')
        if not self.processed_data_dir.exists():
            self.processed_data_dir.mkdir(exist_ok=True, parents=True)
        self.device = device
        self.user2item_source, self.user2item_target, self.num_source, self.num_target = self.load_data(dataset)
        self.align_user()
        self.source = source  # name of source domain
        self.target = target  # name of target domain
        # self.source_set=  # todo list of the source items careful 没用暂时不要
        # self.target_set= # todo list of the target items
        # self.num_users = len() # todo

        # print(f"Finish loading: {self.num_user} users, {self.num_source} source data, {self.num_target} target data")
        print('gen testing dataset ...')
        # call init_test_data for info of evaluation
        # self.init_test_data()
        print("Finish loading ...")
        self.test_count = 0
        self.batch_count = 0
        self.count = 0
        self.epoch = 1

        self.read_data(self.train_path_s)

    def load_data(self, dataset):
        print("Loading data: ", {dataset})
        user2_item_source, _, _, item_num_source = self.read_data(self.train_path_s)
        user2_item_target, _, _, item_num_target = self.read_data(self.train_path_t)
        return user2_item_source, user2_item_target, item_num_source, item_num_target

    def read_data(self, path):
        with path.open() as f:
            line = f.readline()
            data = json.loads(line)
        f.close()
        user_num = len(data)
        item_num = 0
        interactions = []
        for user in range(user_num):
            for item in data[user]:
                interactions.append((user, item))
                item_num = max(item, item_num)
        item_num += 1
        return (data, interactions, user_num, item_num)

    def align_user(self):
        u_t_keys = list(self.user2item_target.keys())
        for u_s in self.user2item_source:
            if u_s not in u_t_keys:
                self.user2item_source.pop(u_s)

        u_s_keys = list(self.user2item_source.keys())
        for u_t in self.user2item_target:
            if u_t not in u_s_keys:
                self.user2item_target.pop(u_t)

    def read_bases(self, path):
        with  path.open() as f:
            line = f.readline()
            bases = json.loads(line)
        f.close()
        [feat_u, feat_v] = bases
        feat_u = np.array(feat_u).astype(np.float32)
        feat_v = np.array(feat_v).astype(np.float32)
        return [feat_u, feat_v]

    def normalize(self, features):
        [f_u, f_v] = features
        f_u_norm = np.linalg.norm(f_u, axis=1, keepdims=True)
        f_v_norm = np.linalg.norm(f_v, axis=1, keepdims=True)
        f_u_norm = np.mean(f_u_norm)
        f_v_norm = np.mean(f_v_norm)
        return [f_u / f_u_norm, f_v / f_v_norm]

    def init_test_data(self):
        # todo implement
        self.source_val, self.source_test, self.source_neg, self.target_val, self.target_test, self.target_neg = self.get_test_data()

    def gen_sample_eval_set(self, domain):
        pass

    def get_test_data(self):
        if (not (self.processed_data_dir / f'{self.source}_val.pkl').is_file() or \
                not (self.processed_data_dir / f'{self.source}_val.pkl').is_file()):
            source_val, source_test, source_neg = self.gen_sample_eval_set('s')
            target_val, target_test, target_neg = self.gen_sample_eval_set('t')

            pickle.dump(source_val, (self.processed_data_dir / f'{self.source}_val.pkl').open('wb'))
            pickle.dump(target_val, (self.processed_data_dir / f'{self.target}_val.pkl').open('wb'))

            pickle.dump(source_test, (self.processed_data_dir / f'{self.source}_test.pkl').open('wb'))
            pickle.dump(target_test, (self.processed_data_dir / f'{self.target}_test.pkl').open('wb'))

            pickle.dump(source_neg, (self.processed_data_dir / f'{self.source}_neg.pkl').open('wb'))
            pickle.dump(target_neg, (self.processed_data_dir / f'{self.target}_neg.pkl').open('wb'))

        else:
            source_val = pickle.load((self.processed_data_dir / f'{self.source}_val.pkl').open('rb'))
            target_val = pickle.load((self.processed_data_dir / f'{self.target}_val.pkl').open('rb'))

            source_test = pickle.load((self.processed_data_dir / f'{self.source}_test.pkl').open('rb'))
            target_test = pickle.load((self.processed_data_dir / f'{self.target}_test.pkl').open('rb'))

            source_neg = pickle.load((self.processed_data_dir / f'{self.source}_neg.pkl').open('rb'))
            target_neg = pickle.load((self.processed_data_dir / f'{self.target}_neg.pkl').open('rb'))
        return source_val, source_test, source_neg, target_val, target_test, target_neg


class DatasetDouban(DatasetSideInfo):
    def __init__(self, BATCH_SIZE, dataset='amazon', source="movie", target='book',
                 processed_data_dir='processed_data', device="cuda"):
        super(DatasetDouban, self).__init__(BATCH_SIZE=BATCH_SIZE, dataset=dataset, source=source, target=target,
                                            processed_data_dir=processed_data_dir, device=device)

    def gen_sample_eval_set(self, domain):
        '''

        :param domain:
        :return:
        '''
        u_items = {}
        user_side_info = {}
        val_item_id = {}
        test_item_id = {}
        neg_item_id = {}

        for i in tqdm(self.user2item_source if domain == "s" else self.user2item_target):
            items = self.user2item_source[i] if domain == "s" else self.user2item_target[i]  # 迭代user 得到items
            items_id = [k[0] for k in items]
            u_items[i] = items_id

            items_reviews = items[0][1]  # [k[1] for k in items][0] #
            user_side_info[i] = items_reviews

            if len(items) >= 2:
                items_choices = shuffle(items, random_state=2020)[:2]
                val_item_id[i] = items_choices[0][0]
                test_item_id[i] = items_choices[1][0]

            else:  # 只有一个就同时用做这个 user 的val & test
                val_item_id[i] = items[0][0]
                test_item_id[i] = items[0][0]

            all_neg_source = list(
                (set(list(range(self.num_source if domain == "s" else self.num_target))) - set(items_id)))
            neg_item_id[i] = shuffle(all_neg_source, random_state=2020)[:99]  # 所有没有交互的选前99个负样本

        return val_item_id, test_item_id, neg_item_id, user_side_info, u_items

    def discrete_si(self, u_si):

        user_side_info_embs = list(u_si.values())
        # lrmodel = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=None,
        #                  copy_x=True, algorithm='auto')

        cluster_ids_x, cluster_centers = kmeans(
            X=torch.from_numpy(np.vstack(user_side_info_embs)).to(self.device), num_clusters=2, distance='euclidean',
            device=self.device
        )
        labels_one_hot = [functional.one_hot(cluster_ids_x[i], 2) for i in range(len(cluster_ids_x))]
        user_side_info_discrete = {}
        for idx, u_id in enumerate(u_si):
            user_side_info_discrete[u_id] = labels_one_hot[idx]

        return user_side_info_discrete


if __name__ == '__main__':
    # dataset = DatasetAmazon(4, "amazon2_review_full", processed_data_dir='../processed_data_rating_le_0_3')
    # dataset.get_part_train_indices("source", 1)
    # dataset.get_part_train_indices("target", 1)
    # dataset = DatasetDouban(4, "book_music_profile_full", processed_data_dir='../processed_data_rating_le_0_3')
    # dataset.get_part_train_indices("source", 1)
    # dataset.get_part_train_indices("target", 1)
    dataset = Dataset(4, dataset="amazon2", source="movie", target="book",
                      processed_data_dir='../processed_data_rating_le_0', )
    # i = 1
    # val = dataset.source_val[i]
    # neg = dataset.source_neg[i]
    # neg_val = np.concatenate([neg, val], 1)
    # from pathlib import Path
    #
    # Path('../processed_data/amazon/source2id.pkl').open('rb')
