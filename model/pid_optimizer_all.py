import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import clip_grad_norm_

# Generate noise
def _generate_noise(noise_multiplier, max_norm, parameter, device):
    if noise_multiplier > 0:
        return torch.normal(
            0,
            noise_multiplier * max_norm,
            parameter.grad.shape,
            device=device,
        )
    return 0.0

class PID_RMSprop(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-2,
                 alpha=0.99,
                 eps=1e-8,
                 vp=0.,
                 vi=0.,
                 vd=0.,
                 centered=False,
                 max_per_sample_grad_norm=1.,
                 noise_multiplier=0,
                 batch_size=1,
                 device="cpu",
                 *args,
                 **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr,
                        alpha=alpha,
                        eps=eps,
                        centered=centered,
                        vp=vp,
                        vi=vi,
                        vd=vd)
        super(PID_RMSprop, self).__init__(params, defaults)
        self.max_per_sample_grad_norm = max_per_sample_grad_norm
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        self.device= device

        # self.param_groups is hidden in the optimizer and **kwargs after calling autoencoder.parameters()
        for group in self.param_groups:
            group['aggregate_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in
                                        group['params']]


    def __setstate__(self, state):
        super(PID_RMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

            # May not be necessary as we can use optimizer.zero_grad() in the loop.

    def zero_microbatch_grad(self):
        super(PID_RMSprop, self).zero_grad()

    def clip_grads_(self):

            # Clip gradients in-place
            params = self.param_groups[0]['params']
            clip_grad_norm_(params, max_norm=self.max_per_sample_grad_norm, norm_type=2)

            # Accumulate gradients
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['aggregate_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data)

            ### Original Implementation below ###
            # total_norm = 0.
            # for group in self.param_groups:
            #     total_norm_params = np.sum([param.grad.data.norm(2).item() ** 2 for param in group['params'] if param.requires_grad])
            #     total_norm += total_norm_params
            # total_norm = total_norm ** .5
            # clip_multiplier = min(self.max_per_sample_grad_norm / (total_norm + 1e-8), 1.)

            # for group in self.param_groups:
            #     for param, accum_grad in zip(group['params'], group['aggregate_grads']):
            #         if param.requires_grad:
            #             accum_grad.add_(param.grad.data.mul(clip_multiplier))

    def zero_grad(self):
        for group in self.param_groups:
            for accum_grad in group['aggregate_grads']:
                if accum_grad is not None:
                    accum_grad.zero_()

    def add_noise_(self):
        for group in self.param_groups:
            for param, accum_grad in zip(group['params'], group['aggregate_grads']):
                if param.requires_grad:
                    # Accumulate gradients
                    param.grad.data = accum_grad.clone()

                    # Add noise and update grads
                    # See: https://github.com/facebookresearch/pytorch-dp/blob/master/torchdp/privacy_engine.py
                    noise = _generate_noise(self.noise_multiplier, self.max_per_sample_grad_norm, param, self.device)
                    param.grad += noise / self.batch_size

                    # See alternative below
                    # param.grad.data.add_(_generate_noise(self.noise_multiplier, self.max_per_sample_grad_norm, param) / self.batch_size)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'RMSprop does not support sparse gradients')
                state = self.state[p]

                alpha = group['alpha']
                vp = group['vp']
                vi = group['vi']
                vd = group['vd']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)
                    if vi > 0:
                        state['i_buffer'] = torch.zeros_like(p.data)
                    if vd > 0:
                        state['d_buffer'] = p.data

                square_avg = state['square_avg']
                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                state['step'] += 1

                if vi > 0:
                    i_buffer = state['i_buffer']
                    i_buffer.add_(p.data)
                else:
                    i_buffer = 0.

                if vd > 0.:
                    d_buffer = state['d_buffer']
                    d_buffer = p.data - d_buffer
                    state['d_buffer'] = p.data
                else:
                    d_buffer = 0.

                controller = vp * p.data + vi * i_buffer + vd * d_buffer
                controller = torch.clamp(controller, -10, 10)
                # grad = grad.add(controller)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(
                        -1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                grad = grad.div(avg)
                grad = grad.add(controller)
                p.data.add_(-group['lr'], grad)

        return loss
