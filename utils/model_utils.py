import random

import torch
import numpy as np
from collections import OrderedDict
from torch import nn
import matplotlib.pyplot as plt

# Utility functions
def toggle_grad(model, requires_grad):  # 重复调用降低冗余
    for p in model.parameters():
        p.requires_grad_(requires_grad)  # note 等价于 p.requires_grad= True 一回事,只是封装成函数


def load_model(model, model_path, strict=False, key_map=None):
    if isinstance(model, dict):
        restore_model_dict = torch.load(model_path)
        for module_tag in restore_model_dict:
            model_state_dict = restore_model_dict[module_tag]
            network = model[module_tag]
            if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
                network = network.module
            load_net_clean = OrderedDict()
            for key, param in model_state_dict.items():
                if key.startswith('module.'):
                    load_net_clean[key[7:]] = param
                else:
                    load_net_clean[key] = param

            network.load_state_dict(load_net_clean, strict=strict)
    elif isinstance(model, nn.Module):
        model_state_dict = torch.load(model_path)
        network = model
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        load_net_clean = OrderedDict()
        for key, param in model_state_dict.items():
            # if key.startswith('gen.'):
            #     load_net_clean[key[4:]] = param
            # el
            if (key_map is not None):
                if key.split('.')[0] in list(key_map.keys()):
                    module_name = "".join(
                        [key_map[key.split('.')[0]]] + [f".{param_name}" for param_name in key.split('.')[1:]])
                    load_net_clean[module_name] = param  # key: 记载的model name map_key[key] = 现有model key
            else:
                if key.startswith('module.'):
                    load_net_clean[key[7:]] = param
                else:
                    load_net_clean[key] = param

        network.load_state_dict(load_net_clean, strict=strict)

    else:
        raise TypeError(f"{model} is not legal type.")


def save_model():
    pass


def use_optimizer(network, params):
    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'sparse_adam':
        optimizer = torch.optim.SparseAdam(network.parameters(), lr=params['lr'])

    elif params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=params['lr'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['lr'],
                                        alpha=params['alpha'],
                                        momentum=params['momentum'])
    return optimizer


def select_negative_items(realData, num_pm_percent=0, num_zr_percent=0):
    '''

    :param realData: n-dim indicator vec spacifying whether u has purchased each item i
    :param num_pm: # of neg items (partial-masking) sampled on the t-th iteration
    :param num_zr: # of neg items (zero@ reconstruction reg) sampled on the t-th iteration
    :return:
    '''
    realData = realData.cpu().numpy()
    data = np.array(realData)

    n_items_pm = np.zeros_like(data)
    n_items_zr = np.zeros_like(data)
    for i in range(data.shape[0]):
        unobserved_items = np.where(data[i] == 0)[0]  # 找出非零的位置, 取出tuple
        non_purchased_length = len(unobserved_items)
        num_pm = int(non_purchased_length * num_pm_percent / 100)
        num_zr = int(non_purchased_length * num_zr_percent / 100)
        n_item_index_pm = np.random.choice(unobserved_items, num_pm, replace=False)
        n_item_index_zr = np.random.choice(unobserved_items, num_zr, replace=False)

        n_items_pm[i][n_item_index_pm] = 1  # 不对因为这样pm zr 两者是不重叠的,本身就是没有关系的 note correct
        n_items_zr[i][n_item_index_zr] = 1

    return n_items_pm, n_items_zr


def plot_loss(loss, save_dir, loss_fig_title='loss during Training'):
    loss_fig = plt.figure(figsize=(10, 5))
    plt.title(loss_fig_title)

    if isinstance(loss, dict):
        for key in loss:
            plt.plot(loss[key], label=key)
    else:
        plt.plot(loss)
    # get plot axis
    ax = plt.gca()
    # remove right and top spine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # add labels and create legend
    plt.xlabel("num_epochs")
    plt.legend()
    plt.show()
    loss_fig.savefig(save_dir / f'{loss_fig_title}.jpg')


def save_config(args):
    pass


def load_config(args):
    pass
    # if args.exp_name is None: # note 这里要自动带入exp_name 而不是找他
    #     pass


if __name__ == '__main__':
    data = np.random.randint(0, 2, (300, 100))
    num_pm = 20
    num_zr = 90
    select_negative_items(data, num_pm, num_zr)
