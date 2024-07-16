from geomloss import SamplesLoss
# from .spp import *
from loss.cl_loss import *
from torch import FloatTensor

from import_torch import *

CLIP_BOUND = 1.
SENSITIVITY = 2.


def master_hook_adder(module, grad_input, grad_output):
    global dynamic_hook_function
    return dynamic_hook_function(module, grad_input, grad_output)


def dummy_hook(module, grad_input, grad_output):
    '''do nothing'''
    return


def modify_gradnorm_data_hook(module, grad_input, grad_output):
    grad_wrt_data = grad_input[1]  # 获取对应的梯度信息 note data 针对的是grad[0]
    grad_input_shape = grad_wrt_data.size()
    batch_size = grad_input.shape[0]
    clip_bound = CLIP_BOUND / batch_size  # 对梯度bound 进行维度平均然后构建norm
    grad_wrt_data = grad_wrt_data.view(batch_size, -1)
    grad_input_norm = torch.norm(grad_wrt_data, p=2, dim=1)

    # clipping gradient
    cliep_coef = clip_bound / (grad_input_norm + 1e-10)  # 利用构建的norm 进行下梯度的clip
    cliep_coef = cliep_coef.unsqueeze(-1)
    grad_wrt_data = cliep_coef * grad_wrt_data
    grad_input_new = [grad_wrt_data.view(grad_input_shape)]
    for i in range(len(grad_input) - 1):
        grad_input_new.append(grad_input[i + 1])
    return tuple(grad_input_new)


def grad_dp_data_hook(module, grad_input, grad_output):
    global noise_multiplier  # todo 怎么赋值
    # print("noise_multiplier in grad_dp_data_hook: ", noise_multiplier)
    # for idx, grad_item in enumerate(grad_input):
    #     print(grad_item.size(), "-" * idx)  #
    #     # note  grad_input 包含三个梯度信息,(bias, batch_data, weight)
    #     '''
    #     torch.Size([1024])
    #     torch.Size([128, 9755]) -
    #     torch.Size([9755, 1024]) --
    #     '''
    grad_wrt_data = grad_input[1]  # grad_wrt_data 在tuple[1]位置
    grad_input_shape = grad_wrt_data.size()
    # print(grad_input_shape, "====================",grad_input)
    batchsize = grad_input_shape[0]
    clip_bound = CLIP_BOUND / batchsize

    grad_wrt_data = grad_wrt_data.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_data, p=2, dim=1)

    # clipping
    cliep_coef = clip_bound / (grad_input_norm + 1e-10)
    clip_coef = torch.min(cliep_coef, torch.ones_like(cliep_coef))
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_data = clip_coef * grad_wrt_data

    # noisy
    noise = clip_bound * noise_multiplier * SENSITIVITY * torch.randn_like(grad_wrt_data)
    # print(grad_wrt_data)
    # print(noise)
    grad_wrt_data = grad_wrt_data + noise  # * 1000000

    grad_input_new = []
    for i in range(len(grad_input)):
        if i == 1:
            grad_input_new.append(grad_wrt_data.view(grad_input_shape))
        else:
            grad_input_new.append(grad_input[i])
    return tuple(grad_input_new, )


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

# note ############################### spectral wgan model ##########################
class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class GeneratorConEmbLS(nn.Module):
    def __init__(self, itemCount, userCount, emb_size):
        '''
        :param itemCount: dim of latent
        :param info_shape: dim of the condition
        '''
        super(GeneratorConEmbLS, self).__init__()
        self.user_embeddings = nn.Embedding(userCount, emb_size)
        self.user_embeddings.weight.data = torch.from_numpy(
            np.random.normal(0, 0.01, size=[userCount, emb_size])).float()
        # self.user_embeddings.weight.requires_grad = False
        self.itemCount = itemCount
        self.gen = nn.Sequential(
            nn.Linear(emb_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Linear(4096, itemCount),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout()

    def forward(self, batch_users):
        '''purchase vec'''
        con_user_embeddings = self.user_embeddings(batch_users)
        input = con_user_embeddings
        result = self.gen(self.dropout(input))
        return (result.squeeze(),)


class DiscriminatorConEmbOTLS(nn.Module):
    def __init__(self, itemCount, userCount, emb_size):
        super(DiscriminatorConEmbOTLS, self).__init__()

        self.user_embeddings = nn.Embedding(userCount, emb_size)
        self.user_embeddings.weight.data = torch.from_numpy(
            np.random.normal(0, 0.01, size=[userCount, emb_size])).float()
        # self.user_embeddings.weight.requires_grad = False
        # self.rating_encoder = nn.Sequential(
        #     nn.Linear(itemCount, 4096),
        #     nn.ReLU(True),
        #     nn.Linear(4096, emb_size * 4)
        # )
        self.encoder = nn.ModuleList([
            SpectralNorm(nn.Linear(itemCount + emb_size, 2048)),  # 会包含三组梯度,
            nn.ReLU(True),
            SpectralNorm(nn.Linear(2048, 1024)),
            nn.ReLU(True),
            SpectralNorm(nn.Linear(1024, 512)),
            nn.ReLU(True),
            SpectralNorm(nn.Linear(512, 256)),
            nn.ReLU(True),
            SpectralNorm(nn.Linear(256, 1))
        ]
            # nn.ReLU(True),
            # nn.Linear(256, 64),
            # nn.ReLU(True),
            # nn.Linear(256, 128),
            # # nn.ReLU(True),
            # # nn.Linear(16, 1),
            # # nn.Sigmoid()
        )

        # self.decoder = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.ReLU(True),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024, 2048),
        #     nn.ReLU(True),
        #     nn.Linear(2048, itemCount),
        # )
        self.dropout = nn.Dropout()

    def forward(self, data, batch_users):
        rating_embeddings = data  # sself.rating_encoder(data)
        con_user_embeddings = self.user_embeddings(batch_users)
        data_c = torch.cat((rating_embeddings, con_user_embeddings), 1)
        x = self.dropout(data_c)
        for enc_layer in self.encoder:
            x = enc_layer(x)
        # latent = x
        # # latent = self.encoder(data_c)
        # latent_act = F.relu(latent)
        # result = self.decoder(latent_act)
        return x  # result, latent


class GeneratorConEmbBigLS(nn.Module):
    def __init__(self, itemCount, userCount, emb_size):
        '''
        :param itemCount: dim of latent
        :param info_shape: dim of the condition
        '''
        super(GeneratorConEmbBigLS, self).__init__()
        self.user_embeddings = nn.Embedding(userCount, emb_size)
        self.user_embeddings.weight.data = torch.from_numpy(
            np.random.normal(0, 0.01, size=[userCount, emb_size])).float()
        # self.user_embeddings.weight.requires_grad = False
        self.itemCount = itemCount
        self.gen = nn.Sequential(
            nn.Linear(emb_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 4096),
            nn.Dropout(),
            nn.ReLU(True),
            # nn.Linear(4096, 8192),
            # nn.ReLU(True),
            # nn.Linear(8192, 16384),
            # nn.ReLU(True),
            nn.Linear(4096, itemCount),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout()

    def forward(self, batch_users):
        '''purchase vec'''
        con_user_embeddings = self.user_embeddings(batch_users)
        input = con_user_embeddings
        result = self.gen(self.dropout(input))
        return (result.squeeze(),)


class DiscriminatorConEmbOTBigLS(nn.Module):
    def __init__(self, itemCount, userCount, emb_size):
        super(DiscriminatorConEmbOTBigLS, self).__init__()

        self.user_embeddings = nn.Embedding(userCount, emb_size)
        self.user_embeddings.weight.data = torch.from_numpy(
            np.random.normal(0, 0.01, size=[userCount, emb_size])).float()
        # self.user_embeddings.weight.requires_grad = False
        # self.rating_encoder = nn.Sequential(
        #     nn.Linear(itemCount, 4096),
        #     nn.ReLU(True),
        #     nn.Linear(4096, emb_size * 4)
        # )
        self.encoder = nn.ModuleList([
            SpectralNorm(nn.Linear(itemCount + emb_size, 4096)),  # 会包含三组梯度,
            # nn.ReLU(True),
            # nn.Linear(16384, 8192),
            # nn.ReLU(True),
            # nn.Linear(8192, 4096),
            nn.ReLU(True),
            SpectralNorm(nn.Linear(4096, 2048)),
            nn.ReLU(True),
            SpectralNorm(nn.Linear(2048, 1024)),
            nn.ReLU(True),
            SpectralNorm(nn.Linear(1024, 512)),
            nn.ReLU(True),
            SpectralNorm(nn.Linear(512, 1))]
            # nn.ReLU(True),
            # nn.Linear(256, 64),
            # nn.ReLU(True),
            # nn.Linear(256, 128),
            # # nn.ReLU(True),
            # # nn.Linear(16, 1),
            # # nn.Sigmoid()
        )

        # self.decoder = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.ReLU(True),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024, 2048),
        #     nn.ReLU(True),
        #     nn.Linear(2048, 4096),
        #     # nn.ReLU(True),
        #     # nn.Linear(4096, 8192),
        #     # nn.ReLU(True),
        #     # nn.Linear(8192, 16384),
        #     nn.ReLU(True),
        #     nn.Linear(4096, itemCount),
        # )
        self.dropout = nn.Dropout()

    def forward(self, data, batch_users):
        rating_embeddings = data  # sself.rating_encoder(data)
        con_user_embeddings = self.user_embeddings(batch_users)
        data_c = torch.cat((rating_embeddings, con_user_embeddings), 1)
        x = self.dropout(data_c)
        for enc_layer in self.encoder:
            x = enc_layer(x)
        # latent = x
        # # latent = self.encoder(data_c)
        # latent_act = F.relu(latent)
        # result = self.decoder(latent_act)
        return x  # result, latent

# note ########################## raw wgan model ################################
class GeneratorConEmb(nn.Module):
    def __init__(self, itemCount, userCount, emb_size):
        '''
        :param itemCount: dim of latent
        :param info_shape: dim of the condition
        '''
        super(GeneratorConEmb, self).__init__()
        self.user_embeddings = nn.Embedding(userCount, emb_size)
        self.user_embeddings.weight.data = torch.from_numpy(
            np.random.normal(0, 0.01, size=[userCount, emb_size])).float()
        # self.user_embeddings.weight.requires_grad = False
        self.itemCount = itemCount
        self.gen = nn.Sequential(
            nn.Linear(emb_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Linear(4096, itemCount),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout()

    def forward(self, batch_users):
        '''purchase vec'''
        con_user_embeddings = self.user_embeddings(batch_users)
        input = con_user_embeddings
        result = self.gen(self.dropout(input))
        return (result.squeeze(),)


class DiscriminatorConEmbOT(nn.Module):
    def __init__(self, itemCount, userCount, emb_size):
        super(DiscriminatorConEmbOT, self).__init__()

        self.user_embeddings = nn.Embedding(userCount, emb_size)
        self.user_embeddings.weight.data = torch.from_numpy(
            np.random.normal(0, 0.01, size=[userCount, emb_size])).float()
        # self.user_embeddings.weight.requires_grad = False
        # self.rating_encoder = nn.Sequential(
        #     nn.Linear(itemCount, 4096),
        #     nn.ReLU(True),
        #     nn.Linear(4096, emb_size * 4)
        # )
        # self.encoder = nn.ModuleList([
        #     nn.Linear(itemCount + emb_size, 2048),  # 会包含三组梯度,
        #     nn.ReLU(True),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(True),
        #     nn.Linear(512, 256),
        #     # nn.ReLU(True),
        #     # nn.Linear(256, 1)
        # ]
        #     # nn.ReLU(True),
        #     # nn.Linear(256, 64),
        #     # nn.ReLU(True),
        #     # nn.Linear(256, 128),
        #     # # nn.ReLU(True),
        #     # # nn.Linear(16, 1),
        #     # # nn.Sigmoid()
        # )
        self.encoder= nn.Sequential(
            nn.Linear(itemCount + emb_size, 2048),  # 会包含三组梯度,
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
        )

        # self.decoder = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.ReLU(True),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024, 2048),
        #     nn.ReLU(True),
        #     nn.Linear(2048, itemCount),
        # )
        self.dropout = nn.Dropout()

    def forward(self, data, batch_users):
        rating_embeddings = data  # sself.rating_encoder(data)
        con_user_embeddings = self.user_embeddings(batch_users)
        data_c = torch.cat((rating_embeddings, con_user_embeddings), 1)
        x = self.dropout(data_c)
        x= self.encoder(x)
        # for enc_layer in self.encoder:
        #     x = enc_layer(x)
        # latent = x
        # # latent = self.encoder(data_c)
        # latent_act = F.relu(latent)
        # result = self.decoder(latent_act)
        return x #result, latent


class GeneratorConEmbBig(nn.Module):
    def __init__(self, itemCount, userCount, emb_size):
        '''
        :param itemCount: dim of latent
        :param info_shape: dim of the condition
        '''
        super(GeneratorConEmbBig, self).__init__()
        self.user_embeddings = nn.Embedding(userCount, emb_size)
        self.user_embeddings.weight.data = torch.from_numpy(
            np.random.normal(0, 0.01, size=[userCount, emb_size])).float()
        # self.user_embeddings.weight.requires_grad = False
        self.itemCount = itemCount
        self.gen = nn.Sequential(
            nn.Linear(emb_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 4096),
            nn.Dropout(),
            nn.ReLU(True),
            # nn.Linear(4096, 8192),
            # nn.ReLU(True),
            # nn.Linear(8192, 16384),
            # nn.ReLU(True),
            nn.Linear(4096, itemCount),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout()

    def forward(self, batch_users):
        '''purchase vec'''
        con_user_embeddings = self.user_embeddings(batch_users)
        input = con_user_embeddings
        result = self.gen(self.dropout(input))
        return (result.squeeze(),)


class DiscriminatorConEmbOTBig(nn.Module):
    def __init__(self, itemCount, userCount, emb_size):
        super(DiscriminatorConEmbOTBig, self).__init__()

        self.user_embeddings = nn.Embedding(userCount, emb_size)
        self.user_embeddings.weight.data = torch.from_numpy(
            np.random.normal(0, 0.01, size=[userCount, emb_size])).float()
        # self.user_embeddings.weight.requires_grad = False
        # self.rating_encoder = nn.Sequential(
        #     nn.Linear(itemCount, 4096),
        #     nn.ReLU(True),
        #     nn.Linear(4096, emb_size * 4)
        # )
        self.encoder = nn.Sequential(
            nn.Linear(itemCount + emb_size, 4096),  # 会包含三组梯度,
            nn.ReLU(True),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512)
        )
        # self.encoder = nn.ModuleList([
        #     nn.Linear(itemCount + emb_size, 4096),  # 会包含三组梯度,
        #     # nn.ReLU(True),
        #     # nn.Linear(16384, 8192),
        #     # nn.ReLU(True),
        #     # nn.Linear(8192, 4096),
        #     nn.ReLU(True),
        #     nn.Linear(4096, 2048),
        #     nn.ReLU(True),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024, 512)]
        #     # nn.ReLU(True),
        #     # nn.Linear(512, 256)]
        #     # nn.ReLU(True),
        #     # nn.Linear(256, 64),
        #     # nn.ReLU(True),
        #     # nn.Linear(256, 128),
        #     # # nn.ReLU(True),
        #     # # nn.Linear(16, 1),
        #     # # nn.Sigmoid()
        # )

        # self.decoder = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.ReLU(True),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024, 2048),
        #     nn.ReLU(True),
        #     nn.Linear(2048, 4096),
        #     # nn.ReLU(True),
        #     # nn.Linear(4096, 8192),
        #     # nn.ReLU(True),
        #     # nn.Linear(8192, 16384),
        #     nn.ReLU(True),
        #     nn.Linear(4096, itemCount),
        # )
        self.dropout = nn.Dropout()

    def forward(self, data, batch_users):
        rating_embeddings = data  # sself.rating_encoder(data)
        con_user_embeddings = self.user_embeddings(batch_users)
        data_c = torch.cat((rating_embeddings, con_user_embeddings), 1)
        x = self.dropout(data_c)
        x = self.encoder(x)
        # for enc_layer in self.encoder:
        #     x = enc_layer(x)
        # latent = x
        # # latent = self.encoder(data_c)
        # latent_act = F.relu(latent)
        # result = self.decoder(latent_act)
        return x #result, latent



class PPGANLS(nn.Module):
    def __init__(self, num_items_s, num_users, emb_size, is_PP, lr_g, lr_d, m, dataset_name,
                 noise_multiplier_):
        super(PPGANLS, self).__init__()
        global noise_multiplier
        noise_multiplier = noise_multiplier_
        self.num_items_s = num_items_s
        self.is_PP = is_PP
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.m = m
        self.num_users = num_users

        # self.generator_s = GeneratorCon(self.num_items_s, emb_size)
        # self.discriminator_s = DiscriminatorConOT(self.num_items_s, emb_size)
        if dataset_name.find("amazon") == -1:  # careful 非豆瓣就是Amazon 没有异常处理
            GModel = GeneratorConEmbLS
            DModel = DiscriminatorConEmbOTLS
        else:
            GModel = GeneratorConEmbBigLS
            DModel = DiscriminatorConEmbOTBigLS


        self.generator_s = GModel(self.num_items_s, self.num_users, emb_size)
        self.discriminator_s = DModel(self.num_items_s, self.num_users, emb_size)

        # all kinds of loss
        self.sink_horn_loss = SamplesLoss()
        self.rec_loss = nn.MSELoss()  # careful 为什么是一个回归任务
        # self.rec_loss = nn.BCELoss()  # careful 为什么是一个回归任务
        # pos_weight = FloatTensor([1.0])  # , device=device)
        # pos_weight = pos_weight.to(args.device)
        # self.rec_loss = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        # self.rec_loss = nn.BCEWithLogitsLoss(reduction="none")

        # self.opt_g = torch.optim.Adam(
        #     params=self.generator_s.parameters(),
        #     lr=self.lr_g,
        #     betas=(0.5, 0.999),
        #     weight_decay=0.0001
        # )
        #
        # self.opt_d = torch.optim.Adam(
        #     params=self.discriminator_s.parameters(),
        #     lr=self.lr_d,
        #     betas=(0.5, 0.999),
        #     weight_decay=0.0001
        # )

    def apply_register_hook(self):
        print('tata')
        global dynamic_hook_function
        # register hook
        self.discriminator_s.encoder[0].register_backward_hook(
            master_hook_adder)  # careful 是encoder还是的decoder

    def train_G_s(self, batch_user_embed, batch_real_user_histories):
        global dynamic_hook_function
        if self.is_PP:
            dynamic_hook_function = grad_dp_data_hook
        else:
            dynamic_hook_function = modify_gradnorm_data_hook
        # careful  不同于cfgan 加上mask表现如何,会不会很受全一向量的影响,以及解决方案

        batch_fake_user_histories = self.generator_s(batch_user_embed)[0]

        # batch_user_embed = self.user_embeddings(batch_user)
        label_real = torch.ones_like(batch_user_embed).float()
        g_loss = self.discriminator_s(batch_fake_user_histories, batch_user_embed)
        # g_fake_recon, g_fake_latent = self.discriminator_s(batch_fake_user_histories, batch_user_embed)
        # g_loss = self._calculate_g_loss(batch_fake_user_histories, g_fake_recon, g_real_latent, g_fake_latent)
        # g_loss = g_loss.mean().view(1)
        # g_loss= torch.norm(g_loss - label_real, p=2,dim =-1)
        g_loss = self.rec_loss(g_loss, label_real)
        return g_loss

    def train_D_s(self, batch_user_embed, batch_real_user_histories):
        # todo is_PP 设置成员变量还是保留现在这种传参形式

        # batch_users_embed = self.user_embeddings(batch_user_embed)
        # careful  不同于cfgan 加上mask表现如何,会不会很受全一向量的影响,以及解决方案
        # on fake data w/o gradient calculation
        with torch.no_grad():
            batch_fake_user_histories = self.generator_s(batch_user_embed)[0]

        if self.is_PP:
            global dynamic_hook_function
            dynamic_hook_function = dummy_hook

        label_real = torch.ones_like(batch_user_embed).float()
        label_fake = torch.zeros_like(batch_user_embed).float()
        d_real = self.discriminator_s(batch_real_user_histories, batch_user_embed)
        d_fake = self.discriminator_s(batch_fake_user_histories, batch_user_embed)

        d_real_loss = self.rec_loss(d_real, label_real)
        d_fake_loss = self.rec_loss(d_fake, label_fake)
        # d_real =  d_real.mean().view(1)
        # # d_fake= d_fake.mean().view(1)
        # d_loss=d_fake - d_real
        # w_distance=  d_real - d_fake
        # d_loss = self._calculate_d_loss(batch_real_user_histories, batch_fake_user_histories, d_real_recon,
        #                                 d_fake_recon)  # todo check

        return d_real_loss, d_fake_loss

    def forward(self, batch_user_embed):
        ratings = self.generator_s(batch_user_embed)
        return ratings

    def _calculate_d_loss(self, real, fake, real_recon, fake_recon):

        loss_real_recon = self.rec_loss(real_recon, real).mean()
        loss_fake_recon = self.rec_loss(fake_recon, fake).mean()
        d_loss = loss_real_recon + torch.maximum(torch.tensor(0., device=fake.device),
                                                 self.m * loss_real_recon - loss_fake_recon)
        # # loss_align = self.sink_hor
        # n_loss(real_latent, fake_latent)
        # d_loss = loss_recon + loss_align
        return d_loss

    def _calculate_g_loss(self, fake, fake_recon, real_latent, fake_latent):
        '''
        careful 救命这个问题, 训练的时候ground truth 怎么设计 urgent
        :param fake:
        :param fake_recon:
        :param real_latent:
        :param fake_latent:
        :return:
        '''
        loss_align = self.sink_horn_loss(real_latent, fake_latent)
        loss_rec = self.rec_loss(fake_recon, fake).mean()
        g_loss = loss_align * 1000 + loss_rec
        # g_fake_loss = torch.mean(F.binary_cross_entropy_with_logits(fake_, torch.ones_like(fake_)))
        # zr_loss = torch.mean(torch.sum(torch.square(fake - 0) * zr_mask, 1, keepdim=True))
        g_loss = g_loss  # + self.zr_coefficient * zr_loss  # + self.g_reg * self.g_ls weight decay implemented by torch

        return g_loss


class PPGAN(nn.Module):
    def __init__(self, num_items_s, num_users, emb_size, is_PP, lr_g, lr_d, m, dataset_name,noise_multiplier_):
        super(PPGAN, self).__init__()
        self.num_items_s = num_items_s
        self.is_PP = is_PP
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.m = m
        self.num_users = num_users
        global noise_multiplier
        noise_multiplier = noise_multiplier_
        # self.generator_s = GeneratorCon(self.num_items_s, emb_size)
        # self.discriminator_s = DiscriminatorConOT(self.num_items_s, emb_size)
        if dataset_name.find("amazon") == -1:  # careful 非豆瓣就是Amazon 没有异常处理
            GModel = GeneratorConEmb
            DModel = DiscriminatorConEmbOT
        else:
            GModel = GeneratorConEmbBig
            DModel = DiscriminatorConEmbOTBig
        self.generator_s = GModel(self.num_items_s, self.num_users, emb_size)
        self.discriminator_s = DModel(self.num_items_s, self.num_users, emb_size)

        # all kinds of loss
        self.sink_horn_loss = SamplesLoss()
        self.rec_loss = nn.MSELoss()  # careful 为什么是一个回归任务
        # self.rec_loss = nn.BCELoss()  # careful 为什么是一个回归任务
        # pos_weight = FloatTensor([1.0])  # , device=device)
        # pos_weight = pos_weight.to(args.device)
        # self.rec_loss = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        # self.rec_loss = nn.BCEWithLogitsLoss(reduction="none")

        # self.opt_g = torch.optim.Adam(
        #     params=self.generator_s.parameters(),
        #     lr=self.lr_g,
        #     betas=(0.5, 0.999),
        #     weight_decay=0.0001
        # )
        #
        # self.opt_d = torch.optim.Adam(
        #     params=self.discriminator_s.parameters(),
        #     lr=self.lr_d,
        #     betas=(0.5, 0.999),
        #     weight_decay=0.0001
        # )
    def apply_register_hook(self):
        global dynamic_hook_function
        # register hook
        self.discriminator_s.encoder[0].register_backward_hook(
            master_hook_adder)  # careful 是encoder还是的decoder

    def train_G_s(self, batch_user_embed, batch_real_user_histories):
        global dynamic_hook_function
        if self.is_PP:
            dynamic_hook_function = grad_dp_data_hook
        else:
            dynamic_hook_function = modify_gradnorm_data_hook
        # careful  不同于cfgan 加上mask表现如何,会不会很受全一向量的影响,以及解决方案

        batch_fake_user_histories = self.generator_s(batch_user_embed)[0]

        # batch_user_embed = self.user_embeddings(batch_user)

        g_loss = self.discriminator_s(batch_fake_user_histories, batch_user_embed)
        # g_fake_recon, g_fake_latent = self.discriminator_s(batch_fake_user_histories, batch_user_embed)
        # g_loss = self._calculate_g_loss(batch_fake_user_histories, g_fake_recon, g_real_latent, g_fake_latent)
        g_loss = g_loss.mean().view(1)
        return g_loss

    def train_D_s(self, batch_user_embed, batch_real_user_histories):

        # batch_users_embed = self.user_embeddings(batch_user_embed)
        # careful  不同于cfgan 加上mask表现如何,会不会很受全一向量的影响,以及解决方案
        # on fake data w/o gradient calculation
        with torch.no_grad():
            batch_fake_user_histories = self.generator_s(batch_user_embed)[0]
        if self.is_PP:
            global dynamic_hook_function
            dynamic_hook_function = dummy_hook

        d_real = self.discriminator_s(batch_real_user_histories, batch_user_embed)
        d_fake = self.discriminator_s(batch_fake_user_histories, batch_user_embed)
        d_real =  d_real.mean().view(1)
        d_fake= d_fake.mean().view(1)
        d_loss=d_fake - d_real
        w_distance=  d_real - d_fake
        # d_loss = self._calculate_d_loss(batch_real_user_histories, batch_fake_user_histories, d_real_recon,
        #                                 d_fake_recon)  # todo check

        return d_loss, d_real, d_fake , w_distance

    def forward(self, batch_user_embed):
        ratings = self.generator_s(batch_user_embed)
        return ratings

    def _calculate_d_loss(self, real, fake, real_recon, fake_recon):

        loss_real_recon = self.rec_loss(real_recon, real).mean()
        loss_fake_recon = self.rec_loss(fake_recon, fake).mean()
        d_loss = loss_real_recon + torch.maximum(torch.tensor(0., device=fake.device),
                                                 self.m * loss_real_recon - loss_fake_recon)
        # # loss_align = self.sink_hor
        # n_loss(real_latent, fake_latent)
        # d_loss = loss_recon + loss_align
        return d_loss

    def _calculate_g_loss(self, fake, fake_recon, real_latent, fake_latent):
        '''
        careful 救命这个问题, 训练的时候ground truth 怎么设计 urgent
        :param fake:
        :param fake_recon:
        :param real_latent:
        :param fake_latent:
        :return:
        '''
        loss_align = self.sink_horn_loss(real_latent, fake_latent)
        loss_rec = self.rec_loss(fake_recon, fake).mean()
        g_loss = loss_align * 1000 + loss_rec
        # g_fake_loss = torch.mean(F.binary_cross_entropy_with_logits(fake_, torch.ones_like(fake_)))
        # zr_loss = torch.mean(torch.sum(torch.square(fake - 0) * zr_mask, 1, keepdim=True))
        g_loss = g_loss  # + self.zr_coefficient * zr_loss  # + self.g_reg * self.g_ls weight decay implemented by torch

        return g_loss



class SMLwoEmb(nn.Module):
    def __init__(self, num_users, hidden_dim, num_preds, lamda, gamma, default_margin=1):
        '''

        :param num_users:
        :param hidden_dim:
        :param num_preds:
        :param neg_samples:
        :param lamda: param for reg loss vv-
        :param gamma: param for reg bias weights
        '''
        super(SMLwoEmb, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_users = num_users
        self.num_preds = num_preds
        # self.neg_samples = neg_samples
        self.gamma = gamma
        self.lamda = lamda
        self.init = 1 / (self.hidden_dim ** 0.5)
        self.default_margin = default_margin

        # U = torch.randn((num_users, hidden_dim)) * self.init  # self.init --> the std_init
        # P = torch.randn((num_preds, hidden_dim)) * self.init  # self.init --> the std_init
        self.bias = nn.Embedding(num_users, 1)
        self.bias.weight.data = torch.ones(num_users, 1)
        self.pbias = nn.Embedding(num_preds, 1)
        self.bias.weight.data = torch.ones(num_preds, 1)
        # self.user_embeddings = nn.Embedding(num_users, hidden_dim)
        # self.user_embeddings.weight.data = U
        # self.pred_embeddings = nn.Embedding(num_preds, hidden_dim)
        # self.pred_embeddings.weight.data = P

    def forward(self, batch_users_embed, batch_preds_embed, batch_neg_samples_embed, batch_users=None,
                batch_preds=None):
        # user_embed = self.user_embeddings(batch_users)
        # pred_embed = self.pred_embeddings(batch_preds)
        # neg_pred_embed = self.pred_embeddings(batch_neg_samples)
        batch_size, feature_dim = batch_users_embed.shape
        if batch_users is not None:  # default is all ones* default_margin
            bias = self.bias(batch_users)
        else:
            bias = self.ones(batch_users_embed.shape[0], 1) * self.default_margin
        if batch_preds is not None:
            pbias = self.pbias(batch_preds)
        else:
            pbias = self.ones(batch_preds_embed.shape[0], 1) * self.default_margin

        pred_distance = torch.sum(torch.square(batch_users_embed - batch_preds_embed), dim=1)
        pred_distance_neg = torch.sum(
            torch.multiply(batch_users_embed.view(batch_size, 1, feature_dim) - batch_neg_samples_embed,
                           batch_users_embed.view(batch_size, 1, feature_dim) - batch_neg_samples_embed),
            dim=-1)
        pred_distance_PN = torch.sum(
            torch.multiply(batch_preds_embed.view(-1, 1, feature_dim) - batch_neg_samples_embed,
                           batch_preds_embed.view(-1, 1, feature_dim) - batch_neg_samples_embed),
            dim=-1)

        a = torch.maximum(pred_distance.view(-1, 1) - pred_distance_neg + bias, torch.tensor(0.))
        b = torch.maximum(pred_distance.view(-1, 1) - pred_distance_PN + pbias, torch.tensor(0.))

        loss = torch.sum(a) + self.lamda * torch.sum(b)
        loss -= self.gamma * (torch.mean(bias) + torch.mean(pbias))

        #  tf.add_to_collection('user_embedding', user_embedding)
        #
        #  tf.add_to_collection('prd_embedding', prd_embedding)
        #
        #  self.clip_U = tf.assign(U, tf.clip_by_norm(U, 1.0, axes=[1]))
        #  self.clip_P = tf.assign(P, tf.clip_by_norm(P, 1.0, axes=[1]))
        #  self.clip_B = tf.assign(B, tf.clip_by_value(B, 0, 1.0))
        #  self.clip_B1 = tf.assign(B1, tf.clip_by_value(B1, 0, 1.0))

        return loss


class SML(nn.Module):
    def __init__(self, num_users, hidden_dim, num_preds, neg_samples, lamda, gamma):
        super(SML, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_users = num_users
        self.num_preds = num_preds
        self.neg_samples = neg_samples
        self.gamma = gamma
        self.lamda = lamda
        self.init = 1 / (self.hidden_dim ** 0.5)

        U = torch.randn((num_users, hidden_dim)) * self.init  # self.init --> the std_init
        P = torch.randn((num_preds, hidden_dim)) * self.init  # self.init --> the std_init
        self.bias = nn.Embedding(num_users, 1)
        self.bias.weight.data = torch.ones(num_users, 1)
        self.pbias = nn.Embedding(num_preds, 1)
        self.bias.weight.data = torch.ones(num_preds, 1)
        self.user_embeddings = nn.Embedding(num_users, hidden_dim)
        self.user_embeddings.weight.data = U
        self.pred_embeddings = nn.Embedding(num_preds, hidden_dim)
        self.pred_embeddings.weight.data = P

    def forward(self, batch_users, batch_preds, batch_neg_samples):
        user_embed = self.user_embeddings(batch_users)
        pred_embed = self.pred_embeddings(batch_preds)
        neg_pred_embed = self.pred_embeddings(batch_neg_samples)
        bias = self.bias(batch_users)
        pbias = self.pbias(batch_preds)

        pred_distance = torch.sum(torch.square(user_embed - pred_embed), dim=1)
        pred_distance_neg = torch.sum(torch.multiply(user_embed - neg_pred_embed, user_embed - neg_pred_embed), dim=1)
        pred_distance_PN = torch.sum(torch.multiply(pred_embed - neg_pred_embed, pred_embed - neg_pred_embed), dim=1)

        a = torch.maximum(pred_distance - pred_distance_neg + bias, torch.tensor(0.))
        b = torch.maximum(pred_distance - pred_distance_PN + pbias, torch.tensor(0.))

        loss = torch.sum(a) + self.lamda * torch.sum(b)
        loss -= self.gamma * (torch.mean(bias) + torch.mean(pbias))

        #  tf.add_to_collection('user_embedding', user_embedding)
        #
        #  tf.add_to_collection('prd_embedding', prd_embedding)
        #
        #  self.clip_U = tf.assign(U, tf.clip_by_norm(U, 1.0, axes=[1]))
        #  self.clip_P = tf.assign(P, tf.clip_by_norm(P, 1.0, axes=[1]))
        #  self.clip_B = tf.assign(B, tf.clip_by_value(B, 0, 1.0))
        #  self.clip_B1 = tf.assign(B1, tf.clip_by_value(B1, 0, 1.0))

        return loss


class InfomaxD(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(InfomaxD, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + z_dim, (x_dim + z_dim) // 2),
            nn.ReLU(True),
            nn.Linear((x_dim + z_dim) // 2, (x_dim + z_dim) // 4),
            nn.ReLU(True),
            nn.Linear((x_dim + z_dim) // 4, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )

    def forward(self, x, z):
        x = torch.cat((x, z), -1)
        return self.net(x)


class DMF(nn.Module):
    def __init__(self, num_users, num_items, emb_size, is_sparse=False):
        super(DMF, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, emb_size, sparse=is_sparse)
        self.user_embeddings.weight.data = torch.from_numpy(
            np.random.normal(0, 0.01, size=[num_users, emb_size])).float()

        self.item_embeddings = nn.Embedding(num_items, emb_size, sparse=is_sparse)
        self.item_embeddings.weight.data = torch.from_numpy(
            np.random.normal(0, 0.01, size=[num_items, emb_size])).float()
        self.num_items = num_items

    # def forward(self, batch_users):
    #     user_emb = self.user_embeddings(batch_users)
    #     allitem = torch.LongTensor(np.array(range(self.num_items))).to(batch_users.device)
    #     out_user = user_emb.unsqueeze(1)
    #     out_item = self.item_embeddings(allitem)
    #     preds = torch.sum(out_user * out_item, dim=2)
    #     return preds, user_emb

    def forward(self, batch_users, h_users_s):  # careful 利用了user_embedding_s 信息,效果有提升
        user_emb = self.user_embeddings(batch_users)
        user_emb = torch.div(user_emb.add(h_users_s), 2.0)
        allitem = torch.LongTensor(np.array(range(self.num_items))).to(batch_users.device)
        out_user = user_emb.unsqueeze(1)
        out_item = self.item_embeddings(allitem)
        preds = torch.sum(out_user * out_item, dim=2)
        return preds, user_emb, out_item


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, emb_size, dropout):
        super(AutoEncoder, self).__init__()
        # self.encoder_s = nn.Sequential(
        #     nn.Linear(input_dim, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, emb_size),
        # )

        self.encoder_s = nn.Sequential(
            nn.Linear(input_dim, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
        )
        # self.encoder_s = nn.Sequential(
        #     # nn.Linear(input_dim, 8192),
        #     # nn.ReLU(),
        #     # nn.Linear(8192, 4096),
        #     # nn.ReLU(True),
        #     nn.Linear(input_dim, 2048),
        #     nn.ReLU(True),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024, emb_size)
        # )
        # nn.Linear(256, 512),
        # nn.ReLU(True),
        # nn.Linear(512, 1024),
        # nn.ReLU(True),
        # nn.Linear(1024, 2048),
        # nn.ReLU(True),
        # nn.Linear(2048, 4096),
        # nn.ReLU(True),
        # nn.Linear(4096, 8192),
        # nn.ReLU(True),
        # nn.Linear(8192, 16384),
        self.dropout = nn.Dropout(dropout)
        # self.decoder_s = nn.Sequential(
        #     nn.Linear(emb_size, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, input_dim),
        # )

        self.decoder_s = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, input_dim),
        )

        # self.decoder_s = nn.Sequential(
        #     nn.Linear(emb_size, 1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024, 2048),
        #     # nn.ReLU(True),
        #     # nn.Linear(2048, 4096),
        #     # nn.ReLU(True),
        #     # nn.Linear(4096, 8192),
        #     nn.ReLU(),
        #     nn.Linear(2048, input_dim)
        # )

    def forward(self, batch_ratings):
        z = self.encoder_s(self.dropout(batch_ratings))
        features_z = F.relu(z)
        rec_ratings = self.decoder_s(features_z)
        return rec_ratings, z


class AutoEncoderBig(nn.Module):
    def __init__(self, input_dim, emb_size, dropout):
        super(AutoEncoderBig, self).__init__()
        # self.encoder_s = nn.Sequential(
        #     nn.Linear(input_dim, emb_size),
        #     nn.ReLU(),
        #     nn.Linear(emb_size, emb_size),
        # )
        self.encoder_s = nn.Sequential(
            # nn.Linear(input_dim, 8192),
            # nn.ReLU(),
            # nn.Linear(8192, 4096),
            # nn.ReLU(True),
            nn.Linear(input_dim, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, emb_size)
        )
        # nn.Linear(256, 512),
        # nn.ReLU(True),
        # nn.Linear(512, 1024),
        # nn.ReLU(True),
        # nn.Linear(1024, 2048),
        # nn.ReLU(True),
        # nn.Linear(2048, 4096),
        # nn.ReLU(True),
        # nn.Linear(4096, 8192),
        # nn.ReLU(True),
        # nn.Linear(8192, 16384),
        self.dropout = nn.Dropout(dropout)
        # self.decoder_s = nn.Sequential(
        #     nn.Linear(emb_size, emb_size),
        #     nn.ReLU(),
        #     nn.Linear(input_dim, emb_size),
        # )

        self.decoder_s = nn.Sequential(
            nn.Linear(emb_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            # nn.ReLU(True),
            # nn.Linear(2048, 4096),
            # nn.ReLU(True),
            # nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(2048, input_dim)
        )

    def forward(self, batch_ratings):
        z = self.encoder_s(self.dropout(batch_ratings))
        features_z = F.relu(z)
        rec_ratings = self.decoder_s(features_z)
        return rec_ratings, z


class HeteroCDR(nn.Module):
    def __init__(self, num_users, num_items_s, num_items_t, emb_size, dropout, is_sparse, dataset_name):
        super(HeteroCDR, self).__init__()
        if dataset_name.find("amazon") != -1:
            Encoder = AutoEncoderBig
        else:
            Encoder = AutoEncoder

        self.model_s = Encoder(num_items_s, emb_size, dropout)
        self.model_t = DMF(num_users, num_items_t, emb_size, is_sparse)

        self.orthogonal_w = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(emb_size, emb_size).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

        self.reg_decay_bpr = 0.05  # urgent todo

        pos_weight = FloatTensor([1.])  # , device=device)
        self.L_w_bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        self.L_mse = nn.MSELoss(reduction="none")

    def orthogonal_map(self, z_s, z_t):
        mapped_z_s = torch.matmul(z_s, self.orthogonal_w)
        mapped_z_t = torch.matmul(z_t, torch.transpose(self.orthogonal_w, 1, 0))
        return mapped_z_s, mapped_z_t

    def forward(self, batch_user, batch_user_s):
        # align source
        preds_s, user_emb_s = self.model_s(batch_user_s)

        # model target
        preds_t, user_emb_t, item_emb_t = self.model_t(batch_user, user_emb_s)

        # # domain alignment
        # mapped_z_s, mapped_z_t = self.orthogonal_map(user_emb_s, user_emb_t)
        # map_loss = torch.norm(mapped_z_t - user_emb_t) + torch.norm(mapped_z_t - mapped_z_s)

        # # alignment
        # mapped_z_s, mapped_z_t = self.orthogonal_map(h_user_s, user_f)
        # z_s = torch.matmul(mapped_z_s, torch.transpose(self.orthogonal_w, 1, 0))
        # z_t = torch.matmul(mapped_z_t, self.orthogonal_w)
        # z_s_reg_loss = torch.norm(h_user_s - z_s, p=1, dim=1)
        # z_t_reg_loss = torch.norm(user_f - z_t, p=1, dim=1)

        '''返回的preds_s 和preds_t 维度是不一致的,前者投影过已经不完全是原始的user id 信息'''
        return preds_s, preds_t, user_emb_s, user_emb_t, item_emb_t  # , map_loss

    def _compute_bpr(self, user_emb_t, item_emb_t, batch_user_pos_set, batch_user_neg_set):
        loss_val_bpr_LIST = []

        for idx, (batch_user_pos_items, batch_user_ns_items) in enumerate(
                zip(batch_user_pos_set, batch_user_neg_set)):
            pos_item_embed = item_emb_t[batch_user_pos_items, :]
            neg_item_embed = item_emb_t[batch_user_ns_items, :]
            user_embed = user_emb_t[idx].unsqueeze(0)
            # print(user_embed.shape, pos_item_embed.shape, neg_item_embed.shape)
            loss_bpr_item = self.bpr_loss(user_embed, pos_item_embed, neg_item_embed)

            loss_val_bpr_LIST.append(loss_bpr_item)
        loss_bpr = torch.stack(loss_val_bpr_LIST).mean()

        return loss_bpr

    def _compute_uni(self, user_emb_t, item_emb_t, batch_user_pos_set):

        user_embeds = []
        item_embeds = []
        for idx, batch_user_pos_items in enumerate(batch_user_pos_set):
            pos_item_embed = item_emb_t[batch_user_pos_items, :]
            user_embed = user_emb_t[idx].unsqueeze(0)
            user_embeds.append(user_embed.expand((pos_item_embed.shape[0], user_embed.shape[-1])))
            item_embeds.append(pos_item_embed)

        user_embeddings = torch.cat(user_embeds, dim=0)
        item_embeddings = torch.cat(item_embeds, dim=0)
        ui_embeddings = torch.cat((user_embeddings, item_embeddings), dim=-1)
        n_pairs = user_embeddings.shape[0]
        ui_embeddings = ui_embeddings[torch.randperm(n_pairs)]
        user_embeddings, item_embeddings = ui_embeddings.chunk(2, dim=-1)

        loss_val_align_LIST = []
        loss_val_unif_x_LIST = []
        loss_val_unif_y_LIST = []
        for i in range(0, n_pairs, 512):
            u_emb = user_embeddings[i: (i + 512)]
            i_emb = item_embeddings[i: (i + 512)]
            if u_emb.shape[0] == 1:  # note 特殊的trick 因为uniform_LOSS 没有办法算batch ==1 的case
                continue
            loss_align = align_loss(u_emb, i_emb)
            loss_uniform_x, loss_uniform_y = uniform_loss(u_emb), uniform_loss(i_emb)

            loss_val_align_LIST.append(loss_align)
            loss_val_unif_x_LIST.append(loss_uniform_x)
            loss_val_unif_y_LIST.append(loss_uniform_y)
        loss_align = torch.stack(loss_val_align_LIST).mean()
        loss_uni_x = torch.stack(loss_val_unif_x_LIST).mean().log()
        loss_uni_y = torch.stack(loss_val_unif_y_LIST).mean().log()
        loss_uni_tot = loss_align + ((loss_uni_x + loss_uni_y) / 2)
        return loss_uni_tot

    def _compute_bce(self, preds_t, batch_user_ratings_t):
        loss_bce = self.L_w_bce(preds_t, batch_user_ratings_t).mean()
        return loss_bce

    def compute_loss(self, batch_user, batch_user_ratings_s, batch_user_ratings_t, batch_user_pos_set,
                     batch_user_neg_set, loss_method="bpr"):
        preds_s, preds_t, user_emb_s, user_emb_t, item_emb_t = self.forward(batch_user, batch_user_ratings_s)

        uu_loss = torch.norm(user_emb_s - user_emb_t)
        # careful pred_s 仅仅在0-1 但是batch_user_s 稀疏化后的值很大 todo discuss
        loss_s = self.L_mse(preds_s, batch_user_ratings_s).mean()
        if loss_method == 'uni':
            loss_val = self._compute_uni(user_emb_t, item_emb_t, batch_user_pos_set)

        elif loss_method == 'bpr':

            loss_val = self._compute_bpr(user_emb_t, item_emb_t, batch_user_pos_set, batch_user_neg_set)

        elif loss_method == 'bce':
            loss_val = self._compute_bce(preds_t, batch_user_ratings_t)

        else:
            loss_uni = self._compute_uni(user_emb_t, item_emb_t, batch_user_pos_set)
            loss_bpr = self._compute_bpr(user_emb_t, item_emb_t, batch_user_pos_set, batch_user_neg_set)
            loss_bce = self._compute_bce(preds_t, batch_user_ratings_t)
            loss_val = 0.5 * loss_uni + loss_bce  # + loss_bpr

        return loss_val * 100 + uu_loss + loss_s * 100  # + loss_val_unif*100  # + info_loss

    def bpr_loss(self, user_gnn_emb, item_gnn_emb_pos, item_gnn_emb_neg):
        # note 暂时不考虑加这个东西
        batch_size = user_gnn_emb.shape[0]
        u_e = user_gnn_emb
        pos_e = item_gnn_emb_pos
        neg_e = item_gnn_emb_neg  # 输出已经是pooling 后的结果
        # u_e = self.pooling(user_gnn_emb)
        # pos_e = self.pooling(item_gnn_emb_pos)
        # neg_e = self.pooling(item_gnn_emb_neg.view(-1, item_gnn_emb_neg.shape[2], item_gnn_emb_neg.shape[3])).view(
        #     batch_size, self.K, -1)
        pos_scores = torch.sum(torch.mul(u_e, pos_e), dim=1)
        # pos_align_scores = align_loss(u_e, pos_e)
        neg_scores = torch.sum(torch.mul(u_e, neg_e), dim=1)
        # neg_align_scores = align_loss(u_e, neg_e)
        # neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), dim=-1)
        mf_loss = F.logsigmoid(pos_scores - neg_scores).sum()
        # mf_loss = F.logsigmoid(pos_align_scores - neg_align_scores).sum()
        # mf_loss = torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))

        # cul regularizer
        regularizer = (torch.norm(user_gnn_emb, dim=1).pow(2).sum() + torch.norm(item_gnn_emb_pos, dim=1).pow(
            2).sum() + torch.norm(
            item_gnn_emb_neg, dim=1).pow(2).sum())  # / 2s
        emb_loss = (self.reg_decay_bpr * regularizer).mean()  # / batch_size

        # align_scores = align_loss(u_e, pos_e)
        # # uni_scores = uniform_loss(u_e) + uniform_loss(pos_e)
        #
        # print(mf_loss, emb_loss, align_scores, -mf_loss + emb_loss + align_scores)
        return (-mf_loss + emb_loss)  # * 2 + align_scores  # +uni_scores


class PPGCDR(nn.Module):
    def __init__(self, num_users, num_items_s, num_items_t, emb_size, knn_size, items_t_embed, pool_method='sum', K=1,
                 lamda=0., gamma=0., reg_uu=0., pos_weight=1., device="cuda:0"):
        super(PPGCDR, self).__init__()
        self.num_items_s = num_items_s
        self.num_items_t = num_items_t
        self.num_users = num_users
        self.emb_size = emb_size
        self.knn_size = knn_size
        self.pool_method = pool_method
        self.items_t_embed = items_t_embed
        self.K = K  # top k negative samples

        self.gnn_ui = GNN(emb_size, step=2, batch_norm=True, edge_dropout_rate=0.5, dropout=0.5)
        self.gnn_uu = GNN(emb_size, step=2, batch_norm=True, edge_dropout_rate=0.5, dropout=0.5)

        self.user_align = nn.Sequential(
            nn.Linear(emb_size * 2, emb_size),  # emb_size * 2 * (step+1)
            # nn.Softmax(dim=1),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )  # note 用来拼接不同的user embedding信息

        self.item_align = nn.Sequential(
            nn.Linear(emb_size, emb_size),  # emb_size * 2 * (step+1)
            # nn.Softmax(dim=1),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )  # note 用来拼接不同的user embedding信息

        pos_weight = FloatTensor([pos_weight])  # , device=device)
        pos_weight = pos_weight.to(device)
        self.L_w_bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        self.L_bce = nn.BCEWithLogitsLoss(reduction="none")
        self.L_sml = SMLwoEmb(num_users, emb_size, num_items_t, lamda=lamda, gamma=gamma)
        self.infoD = InfomaxD(num_items_s, num_items_t)  # to(device)

        self.reg_uu = reg_uu
        self.reg_decay_bpr = 0.05  # urgent todo

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool_method == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool_method == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool_method == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def permute_dims(self, z):
        # assert z.dim() == 2
        B, _ = z.size()
        perm = torch.randperm(B)  # .to(device)
        perm_z = z[perm]
        return perm_z

    def mi_loss(self, X, Y):
        '''

        :param f_x: 生成的表征1
        :param X: 原始的数据表征2
        :return:
        '''
        D_xy = self.infoD(X, Y)
        y_permuted = self.permute_dims(Y)
        D_x_y = self.infoD(X, y_permuted)
        info_xy_loss = -(D_xy.mean() - torch.exp(D_x_y - 1).mean())
        return info_xy_loss

    def bpr_loss(self, user_gnn_emb, item_gnn_emb_pos, item_gnn_emb_neg):
        # note 暂时不考虑加这个东西
        batch_size = user_gnn_emb.shape[0]
        u_e = user_gnn_emb
        pos_e = item_gnn_emb_pos
        neg_e = item_gnn_emb_neg  # 输出已经是pooling 后的结果
        # u_e = self.pooling(user_gnn_emb)
        # pos_e = self.pooling(item_gnn_emb_pos)
        # neg_e = self.pooling(item_gnn_emb_neg.view(-1, item_gnn_emb_neg.shape[2], item_gnn_emb_neg.shape[3])).view(
        #     batch_size, self.K, -1)
        pos_scores = torch.sum(torch.mul(u_e, pos_e), dim=1)
        # pos_align_scores = align_loss(u_e, pos_e)
        neg_scores = torch.sum(torch.mul(u_e, neg_e), dim=1)
        # neg_align_scores = align_loss(u_e, neg_e)
        # neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), dim=-1)
        mf_loss = F.logsigmoid(pos_scores - neg_scores).sum()
        # mf_loss = F.logsigmoid(pos_align_scores - neg_align_scores).sum()
        # mf_loss = torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))

        # cul regularizer
        regularizer = (torch.norm(user_gnn_emb, dim=1).pow(2).sum() + torch.norm(item_gnn_emb_pos, dim=1).pow(
            2).sum() + torch.norm(
            item_gnn_emb_neg, dim=1).pow(2).sum())  # / 2s
        emb_loss = (self.reg_decay_bpr * regularizer).mean()  # / batch_size

        align_scores = align_loss(u_e, pos_e)
        # uni_scores = uniform_loss(u_e) + uniform_loss(pos_e)

        print(mf_loss, emb_loss, align_scores, -mf_loss + emb_loss + align_scores)
        return (-mf_loss + emb_loss) * 2 + align_scores  # +uni_scores
        # return mf_loss - emb_loss

    def sml_loss(self, user_gnn_emb, item_gnn_emb_pos, item_gnn_emb_neg, batch_users=None,
                 batch_preds=None):
        return self.L_sml(user_gnn_emb, item_gnn_emb_pos, item_gnn_emb_neg, batch_users=batch_users,
                          batch_preds=batch_preds)

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
        # batch_ratings = torch.from_numpy(batch_ratings).to(device)
        return graph

    def compute_loss(self, batch_user, batch_user_ratings_s, batch_user_ratings_t, batch_user_pos_set,
                     batch_user_neg_set, loss_method="bpr", use_uu_reg=False):
        if use_uu_reg:
            batch_user_embed, batch_item_embed, loss_uu_reg = self.forward_generate(batch_user, batch_user_ratings_s,
                                                                                    batch_user_ratings_t,
                                                                                    use_uu_reg=use_uu_reg)
        else:
            batch_user_embed, batch_item_embed = self.forward_generate(batch_user, batch_user_ratings_s,
                                                                       batch_user_ratings_t,
                                                                       use_uu_reg=use_uu_reg)

        # info_loss= self.mi_loss(batch_user_ratings_s, batch_user_ratings_t)
        loss_val_LIST = []
        for idx, (batch_user_pos_items, batch_user_ns_items) in enumerate(zip(batch_user_pos_set, batch_user_neg_set)):

            pos_item_embed = batch_item_embed[batch_user_pos_items, :]
            neg_item_embed = batch_item_embed[batch_user_ns_items, :]
            user_embed = batch_user_embed[idx].unsqueeze(0)
            if loss_method == 'bpr':
                loss_val = self.bpr_loss(user_embed, pos_item_embed, neg_item_embed)
            elif loss_method == 'sml':
                loss_val = self.sml_loss(user_embed, pos_item_embed, neg_item_embed, batch_user,
                                         batch_user_pos_set)
            else:
                batch_ratings = torch.matmul(batch_user_embed,
                                             batch_item_embed.T)  # todo rating 的值波动太大-> note v1 还是矩阵相乘的版本
                loss_val = self.L_bce(batch_ratings, batch_user_ratings_t).sum()

            loss_val_LIST.append(loss_val)

        loss_val = torch.stack(loss_val_LIST).mean()
        if use_uu_reg:
            return loss_val + self.reg_uu * loss_uu_reg
        else:
            return loss_val  # + info_loss

    def forward(self, batch_user, batch_user_ratings_s, batch_user_ratings_t):
        batch_user_embed, batch_item_embed = self.forward_generate(batch_user, batch_user_ratings_s,
                                                                   batch_user_ratings_t)
        batch_ratings = torch.matmul(batch_user_embed, batch_item_embed.T)
        return (batch_ratings,)

    def forward_generate(self, batch_user, batch_user_ratings_s, batch_user_ratings_t, use_uu_reg=False):
        n_batch_users = batch_user.shape[0]
        batch_users_embed = self.user_embeddings(batch_user)

        # batch_neg_embed= self.user_embeddings(batch_user_neg_set)
        # batch_neg_embed= batch_neg_embed.reshape(n_batch_users, self.K, self.emb_size)  # note batch* num_of_neg_candidate* embsize

        all_items = torch.LongTensor(np.array(range(self.num_items_t))).to(batch_users_embed.device)
        all_items_t_embed = self.items_t_embed(all_items)

        # build uu graph
        A_hat_uu = self.get_user_relation_graph(batch_user_ratings_s)  # 学长的意思是这weight当作约束放在下面用
        g_hidden_s_u = self.gnn_uu(batch_users_embed, A_hat_uu)
        g_hidden_s_u = self.pooling(g_hidden_s_u)

        # build ui graph
        A_hat_ui = self.get_sparse_interaction_graph(batch_user_ratings_t)
        batch_ui_embed_t = torch.cat((batch_users_embed, all_items_t_embed), dim=0)
        g_hidden_t = self.gnn_ui(batch_ui_embed_t, A_hat_ui)
        g_hidden_t_u, g_hidden_t_i = g_hidden_t[:n_batch_users, :], g_hidden_t[n_batch_users:, :]

        g_hidden_t_u = self.pooling(g_hidden_t_u)
        g_hidden_t_i = self.pooling(g_hidden_t_i)  # TODO pooling 内容和目的

        # note 不同拼接后提升t性能的us ut 拼接方式
        g_hidden_u = self.user_align(torch.cat((g_hidden_s_u, g_hidden_t_u), dim=1))

        # # g_hidden_t_i = self.item_align(g_hidden_t_i)
        # # g_hidden_t_u = torch.cat((g_hidden_s_u, g_hidden_t_u), dim=1)
        # # g_hidden_u = (g_hidden_s_u + g_hidden_t_u)/2
        # ratings = torch.matmul(g_hidden_u, g_hidden_t_i.T)  # todo rating 的值波动太大-> note v1 还是矩阵相乘的版本
        #

        # pred_loss = self.L_bce(ratings, batch_user_ratings_t).sum()

        if use_uu_reg:
            # user normalization
            repeat_g_hidden_t_u = g_hidden_t_u.repeat(n_batch_users, 1, 1)
            # note 这一段加不加影响不大,保留着跑跑试试 加但是利用lambda控制其是一个很小的值
            norm_diff_matrix = repeat_g_hidden_t_u - g_hidden_t_u.unsqueeze(1)
            norm = torch.matmul(norm_diff_matrix, norm_diff_matrix.transpose(2, 1))
            loss_reg_user = torch.mul(norm, A_hat_uu).mean()  # urgent todo 确认正负符号
            return g_hidden_u, g_hidden_t_i, loss_reg_user
        #
        else:
            return g_hidden_u, g_hidden_t_i
