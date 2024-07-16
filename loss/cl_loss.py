from loss.import_torch import *


def align_loss(x, y, alpha=2):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean()  # .log()


