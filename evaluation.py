import numpy as np
import torch

def eval(preds, topK):
    sort = np.argsort(-preds, axis=1)[:, :topK]
    hr_arr = np.zeros(shape=[sort.shape[0]])
    ndcg_arr = np.zeros(shape=[sort.shape[0]])
    mrr_arr = np.zeros(shape=[sort.shape[0]])

    rows = np.where(sort == 99)[0]
    cols = np.where(sort == 99)[1]

    hr_arr[rows] = 1.
    ndcg_arr[rows] = np.log(2) / np.log(cols + 2.) # target item 在预测中的位置
    mrr_arr[rows] = 1.0 / (cols + 1.0)
    return hr_arr.tolist(), ndcg_arr.tolist(), mrr_arr.tolist()


def test_process(model, train_loader, feed_data, cuda, topK, method ="",dr_target="cdr", domain='s',
                 mode='val', use_ratings=False):
    if dr_target == 'cdr':
        # cross domain rec 系列 --> 验证target的性能提升
        return test_process_cdr(model, train_loader, feed_data, cuda, topK,method= method, mode=mode)
    elif dr_target == 'sdr':
        # single domain rec 系列
        return test_process_sdr(model, train_loader, feed_data, cuda, topK, method= method,mode=mode, domain=domain,
                                use_ratings=use_ratings)
    elif dr_target == 'ddr':
        # dual transfer 系列
        return test_process_ddr(model, train_loader, feed_data, cuda, topK,method= method, mode=mode)
    else:
        raise NotImplementedError(f"eval for {dr_target} is not implemented!!!")


def test_process_cdr(model, train_loader, feed_data, device, topK, method="",mode='val'):
    all_hr_t_list = []
    all_ndcg_t_list = []
    all_mrr_t_list = []

    fts_s = feed_data['fts_s']
    fts_t = feed_data['fts_t']
    if mode == 'val':
        target_neg = feed_data['target_neg']
        target_test = feed_data['target_val']

    elif mode == 'test':
        target_neg = feed_data['target_neg']
        target_test = feed_data['target_test']

    else:
        raise Exception

    model_dict_flag = False
    if isinstance(model, dict):
        model_dict_flag = True

    for batch_id, data in enumerate(train_loader):
        data = data.reshape([-1])
        val_user_arr = data.numpy()

        v_user = torch.LongTensor(val_user_arr).to(device)

        if model_dict_flag:
            if 'gen_s' in model.keys():
                # v_user_emb = model['user_embed'](v_user)
                v_item_s = model['gen_s'](v_user)  # [0]
                v_item_s = v_item_s[0]  # careful 生成的数据一样!
                v_item_t = fts_t[val_user_arr]
                v_item_t = torch.FloatTensor(v_item_t).to(device)
                res = model['model_t'].forward(v_user, v_item_s, v_item_t)
                y_t = res[0]
            elif 'model_s' in model.keys():
                # v_user_emb = model['user_embed'](v_user)
                v_item_s = model['model_s'](v_user)  # [0]
                v_item_s = v_item_s[0]  # careful 生成的数据一样!
                v_item_t = fts_t[val_user_arr]
                v_item_t = torch.FloatTensor(v_item_t).to(device)
                res = model['model_t'].forward(v_user, v_item_s) #, v_item_t)
                y_t = res[1]

        else:
            v_item_s = fts_s[val_user_arr]
            v_item_s = torch.FloatTensor(v_item_s).to(device)
            res = model.forward(v_user, v_item_s)
            y_t = res[1]

        y_t = y_t.detach().cpu().numpy()

        neg_val_t = np.array([target_neg[ele] + [target_test[ele]] for ele in val_user_arr])
        preds_t = np.stack([y_t[tt][neg_val_t[tt]] for tt in range(neg_val_t.shape[0])])
        hr_t_list, ndcg_t_list, mrr_t_list = eval(preds_t, topK)
        all_hr_t_list += hr_t_list
        all_ndcg_t_list += ndcg_t_list
        all_mrr_t_list += mrr_t_list

    avg_hr_t = np.mean(all_hr_t_list)
    avg_ndcg_t = np.mean(all_ndcg_t_list)
    avg_mrr_t = np.mean(all_mrr_t_list)

    return avg_hr_t, avg_ndcg_t, avg_mrr_t


def test_process_sdr( model, train_loader, feed_data, cuda, topK, method="",mode='val', domain='s', use_ratings=False):
    all_hr_t_list = []
    all_ndcg_t_list = []
    all_mrr_t_list = []
    if domain.lower() == 's':
        fts_t = feed_data['fts_s']

        if mode == 'val':
            target_neg = feed_data['source_neg']
            target_test = feed_data['source_val']

        elif mode == 'test':
            target_neg = feed_data['source_neg']
            target_test = feed_data['source_test']

        else:
            raise Exception
    else:
        fts_t = feed_data['fts_t']

        if mode == 'val':
            target_neg = feed_data['target_neg']
            target_test = feed_data['target_val']

        elif mode == 'test':
            target_neg = feed_data['target_neg']
            target_test = feed_data['target_test']

        else:
            raise Exception

    for batch_id, data in enumerate(train_loader):
        data = data.reshape([-1])  # careful temp change
        v_user= data.to(cuda)
        val_user_arr = data.numpy()
        # v_user = torch.LongTensor(val_user_arr).to(cuda)

        if method =="bpr":
            batch_item_list=[]
            for u_idx in v_user:
                temp_u_idx= u_idx.clone().cpu().numpy().tolist()
                temp_item_list=[]
                temp_item_list+=target_neg[temp_u_idx]
                temp_item_list+=[target_test[temp_u_idx]]
                batch_item_list.append(torch.tensor(temp_item_list))

            v_user= v_user.reshape(-1,1)
            item = torch.stack(batch_item_list).to(cuda)
            user= v_user.repeat(1, item.shape[-1]).to(cuda)
            res = model(user, item)
            y_t = res[0]
            y_t = y_t.detach().cpu().numpy()

            neg_val_t = np.array([target_neg[ele] + [target_test[ele]] for ele in val_user_arr])
            preds_t = y_t #np.stack([y_t[tt][neg_val_t[tt]] for tt in range(neg_val_t.shape[0])])
        else:

            if not use_ratings:

                if isinstance(model, dict):
                    # {'user_embed': self.user_embeddings,
                    #  'gen_s': self.model_s,
                    #  'model_t': self.model_t},
                    # v_user_emb = model['user_embed'](v_user)
                    # v_item_s = model['gen_s'](v_user_emb)  # [0]
                    # v_item_s = v_item_s[0][0]  # careful 生成的数据一样!
                    user_embed = model['user_embed'](v_user.reshape(-1, 1))
                    res = model['gen_s'](user_embed)
                    y_t = res[0]
                else:
                    # user_embed = model(v_user.reshape(-1, 1))
                    res = model.forward(v_user.reshape(-1, 1))
                    y_t = res[0]
            else:
                batch_val_user_ratings = torch.FloatTensor(fts_t[val_user_arr]).to(device=cuda)
                res = model.forward(batch_val_user_ratings)
                if len(res) == 3:
                    y_t = res[1] if domain.lower() == 's' else res[2]
                else:
                    y_t = res[0]

            y_t = y_t.detach().cpu().numpy()

            neg_val_t = np.array([target_neg[ele] + [target_test[ele]] for ele in val_user_arr])
            preds_t = np.stack([y_t[tt][neg_val_t[tt]] for tt in range(neg_val_t.shape[0])])
        hr_t_list, ndcg_t_list, mrr_t_list = eval(preds_t, topK)
        all_hr_t_list += hr_t_list
        all_ndcg_t_list += ndcg_t_list
        all_mrr_t_list += mrr_t_list

    avg_hr_t = np.mean(all_hr_t_list)
    avg_ndcg_t = np.mean(all_ndcg_t_list)
    avg_mrr_t = np.mean(all_mrr_t_list)

    return avg_hr_t, avg_ndcg_t, avg_mrr_t


def test_process_sdr_temp(model, train_loader, feed_data, cuda, topK, method="",mode='val', domain='s', use_ratings=False,
                          dr_target=None,
                          alpha=1.):
    all_hr_t_list = []
    all_ndcg_t_list = []
    all_mrr_t_list = []
    if domain.lower() == 's':
        is_source = True
        fts_t = feed_data['fts_s']

        if mode == 'val':
            target_neg = feed_data['source_neg']
            target_test = feed_data['source_val']

        elif mode == 'test':
            target_neg = feed_data['source_neg']
            target_test = feed_data['source_test']

        else:
            raise Exception
    else:
        is_source = False
        fts_t = feed_data['fts_t']

        if mode == 'val':
            target_neg = feed_data['target_neg']
            target_test = feed_data['target_val']

        elif mode == 'test':
            target_neg = feed_data['target_neg']
            target_test = feed_data['target_test']

        else:
            raise Exception

    for batch_id, data in enumerate(train_loader):
        data = data.reshape([-1])  # careful temp change
        val_user_arr = data.numpy()
        v_user = torch.LongTensor(val_user_arr).to(cuda)
        batch_val_user_ratings = torch.FloatTensor(fts_t[val_user_arr]).to(device=cuda)

        res = model.forward(batch_val_user_ratings, alpha, is_source)
        if len(res) == 2 or domain.lower() == 's':
            y_t = res[1]
        else:
            y_t = res[2]

        y_t = y_t.detach().cpu().numpy()

        neg_val_t = np.array([target_neg[ele] + [target_test[ele]] for ele in val_user_arr])
        preds_t = np.stack([y_t[tt][neg_val_t[tt]] for tt in range(neg_val_t.shape[0])])
        hr_t_list, ndcg_t_list, mrr_t_list = eval(preds_t, topK)
        all_hr_t_list += hr_t_list
        all_ndcg_t_list += ndcg_t_list
        all_mrr_t_list += mrr_t_list

    avg_hr_t = np.mean(all_hr_t_list)
    avg_ndcg_t = np.mean(all_ndcg_t_list)
    avg_mrr_t = np.mean(all_mrr_t_list)

    return avg_hr_t, avg_ndcg_t, avg_mrr_t


def test_process_ddr(model, train_loader, feed_data, cuda, topK,method="",
                     mode='val'):
    '''todo ddtcdr 任务的eval 需要做一些改动在这里实现'''
    all_hr_s_list = []
    all_ndcg_s_list = []
    all_mrr_s_list = []
    all_hr_t_list = []
    all_ndcg_t_list = []
    all_mrr_t_list = []
    fts_s = feed_data['fts_s']
    fts_t = feed_data['fts_t']
    if mode == 'val':
        source_neg = feed_data['source_neg']
        source_test = feed_data['source_val']
        target_neg = feed_data['target_neg']
        target_test = feed_data['target_val']

    elif mode == 'test':
        source_neg = feed_data['source_neg']
        source_test = feed_data['source_test']
        target_neg = feed_data['target_neg']
        target_test = feed_data['target_test']

    else:
        raise Exception

    for batch_id, data in enumerate(train_loader):
        data = data.reshape([-1])
        val_user_arr = data.numpy()
        v_item_s = fts_s[val_user_arr]
        v_item_t = fts_t[val_user_arr]

        # v_item_t = projection(v_item_t,eps,sp,pmode,et)
        # v_item_t = v_item_t.to(torch.float32)
        # v_item_t = torch.FloatTensor(np.random.rand(v_item_t.shape[0], v_item_t.shape[1]))

        v_user = torch.LongTensor(val_user_arr).to(cuda)
        v_item_s = torch.FloatTensor(v_item_s).to(cuda)
        v_item_t = torch.FloatTensor(v_item_t).to(cuda)

        res = model.forward(v_user, v_item_s, v_item_t)
        # if method_name == "etl":
        #     res = model.forward(v_user, v_item_s, v_item_t)
        # elif method_name == "dpcdr":
        #     res = model.forward(v_user, v_item_s)
        # else:
        #     raise NotImplementedError(f"{method_name} is not correctly implemented!!! NOTICE")
        y_s = res[0]
        y_t = res[1]

        y_s = y_s.detach().cpu().numpy()
        y_t = y_t.detach().cpu().numpy()

        neg_val_s = np.array([source_neg[ele] + [source_test[ele]] for ele in val_user_arr])
        preds_s = np.stack([y_s[ss][neg_val_s[ss]] for ss in range(neg_val_s.shape[0])])
        hr_s_list, ndcg_s_list, mrr_s_list = eval(preds_s, topK)
        all_hr_s_list += hr_s_list
        all_ndcg_s_list += ndcg_s_list
        all_mrr_s_list += mrr_s_list

        neg_val_t = np.array([target_neg[ele] + [target_test[ele]] for ele in val_user_arr])
        preds_t = np.stack([y_t[tt][neg_val_t[tt]] for tt in range(neg_val_t.shape[0])])
        hr_t_list, ndcg_t_list, mrr_t_list = eval(preds_t, topK)
        all_hr_t_list += hr_t_list
        all_ndcg_t_list += ndcg_t_list
        all_mrr_t_list += mrr_t_list

    avg_hr_t = np.mean(all_hr_t_list)
    avg_ndcg_t = np.mean(all_ndcg_t_list)
    avg_mrr_t = np.mean(all_mrr_t_list)

    avg_hr_s = np.mean(all_hr_s_list)
    avg_ndcg_s = np.mean(all_ndcg_s_list)
    avg_mrr_s = np.mean(all_mrr_s_list)

    return avg_hr_s, avg_ndcg_s, avg_mrr_s, avg_hr_t, avg_ndcg_t, avg_mrr_t
