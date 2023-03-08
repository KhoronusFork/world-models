"""
Define MDRNN model, supposed to be used as a world model
on the latent space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal

try:
    import sys
    sys.path.append('../../sigpro/dlf')
    from dlf.model import *
except ImportError:
    print('DLF library not found')

def gmm_loss(batch, mus, sigmas, logpi, reduce=True): # pylint: disable=too-many-arguments
    """ Computes the gmm loss.

    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.

    :args batch: (bs1, bs2, *, fs) torch tensor
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args logpi: (bs1, bs2, *, gs) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited

    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))

    NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
    with fs).
    """
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return - torch.mean(log_prob)
    return - log_prob

class _MDRNNBase(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(
            hiddens, (2 * latents + 1) * gaussians + 2)

    def forward(self, *inputs):
        pass

class MDRNN(_MDRNNBase):
    """ MDRNN model for multi steps forward """
    def __init__(self, latents, actions, hiddens, gaussians, mode:str):
        super().__init__(latents, actions, hiddens, gaussians)
        self.mode = mode
        if mode == 'lstm':
            self.rnn = nn.LSTM(latents + actions, hiddens)
        elif mode == 'dlf':
            print('latents:{} actions:{}'.format(latents, actions))
            print('hiddens:{}'.format(hiddens))
            sys_order = 2
            num_head = 10#1000
            num_layers = 3
            L = 1
            Features = latents + actions
            bidirectional = False
            FeaturesOut = hiddens # 2048
            sys_order_expected = FeaturesOut / (num_head * num_layers) # 2048 FeaturesOut in dreamer
            print('sys_order_expected:{}'.format(sys_order_expected))
            sys_order = int(sys_order_expected)
            print('sys_order:{}'.format(sys_order))
            print('TFNet')
            #self.rnnmodel = TFNet(input_size = Features, sys_order = sys_order, num_head = num_head, output_size = FeaturesOut, period = period, num_layers = num_layers, bidirectional = bidirectional, jitter = jitter)
            if True:
                self.rnn = DLF(input_size = Features, 
                               output_size = FeaturesOut, 
                               num_layers = num_layers, 
                               bidirectional = bidirectional, 
                               block=dict(
                                   type='ActivationBlock',
                                   kwargs=dict(
                                       filter=dict(
                                           type='PolyCoef', 
                                           kwargs = dict(
                                           sys_order = sys_order, 
                                           num_head = num_head)))))

            #count_parameters(self.rnnmodel)
            # Plot generator
            self.do_plot = False
            self.num_iterations_plot = 0
            self.bin_num_iterations_plot = 500
            if self.do_plot:
                self.fig_in, self.line_in, self.line_pred_in = create_plot(num_state = 1, max_len = Features, ylim = [(-3,3)], title = 'Input')
                self.fig, self.line, self.line_pred = create_plot(num_state = 1, max_len = FeaturesOut, ylim = [(-3,3)], title = "TrainModel")
                self.hf = plt.figure()
                self.ha = self.hf.add_subplot(111, projection='3d', title='prediction')
                self.hf1 = plt.figure()
                self.ha1 = self.hf1.add_subplot(111, projection='3d', title='input')
                self.hf2 = plt.figure()
                self.ha2 = self.hf2.add_subplot(111, projection='3d', title='latent')
        else:
            print('mdrnn: mode uknown:{}'.format(mode))

    def forward(self, actions, latents): # pylint: disable=arguments-differ
        """ MULTI STEPS forward.

        :args actions: (SEQ_LEN, BSIZE, ASIZE) torch tensor
        :args latents: (SEQ_LEN, BSIZE, LSIZE) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)
        if self.mode == 'lstm':
            outs, _ = self.rnn(ins)
        else:
            outs = self.rnn(ins)

        gmm_outs = self.gmm_linear(outs)

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        rs = gmm_outs[:, :, -2]

        ds = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, rs, ds

class MDRNNCell(_MDRNNBase):
    """ MDRNN model for one step forward """
    def __init__(self, latents, actions, hiddens, gaussians, mode:str):
        super().__init__(latents, actions, hiddens, gaussians)
        self.mode = mode
        if mode == 'lstm':
            self.rnn = nn.LSTMCell(latents + actions, hiddens)
        elif mode == 'dlf':
            print('latents:{} actions:{}'.format(latents, actions))
            print('hiddens:{}'.format(hiddens))
            sys_order = 2
            num_head = 10#1000
            num_layers = 3
            L = 1
            Features = latents + actions
            bidirectional = False
            FeaturesOut = hiddens # 2048
            sys_order_expected = FeaturesOut / (num_head * num_layers) # 2048 FeaturesOut in dreamer
            print('sys_order_expected:{}'.format(sys_order_expected))
            sys_order = int(sys_order_expected)
            print('sys_order:{}'.format(sys_order))
            print('TFNet')
            self.dlf_sys_order = sys_order
            self.dlf_num_head = 10#1000
            #self.rnnmodel = TFNet(input_size = Features, sys_order = sys_order, num_head = num_head, output_size = FeaturesOut, period = period, num_layers = num_layers, bidirectional = bidirectional, jitter = jitter)
            if True:
                self.rnn = DLF(input_size = Features, 
                               output_size = FeaturesOut, 
                               num_layers = num_layers, 
                               bidirectional = bidirectional, 
                               block=dict(
                                   type='ActivationBlock',
                                   kwargs=dict(
                                       filter=dict(
                                           type='PolyCoef', 
                                           kwargs = dict(
                                           sys_order = sys_order, 
                                           num_head = num_head)))))
        else:
            print('mdrnn: mode uknown:{}'.format(mode))

    def forward(self, action, latent, hidden): # pylint: disable=arguments-differ
        """ ONE STEP forward.

        :args actions: (BSIZE, ASIZE) torch tensor
        :args latents: (BSIZE, LSIZE) torch tensor
        :args hidden: (BSIZE, RSIZE) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
        the GMM prediction for the next latent, gaussian prediction of the
        reward, logit prediction of terminality and next hidden state.
            - mu_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (BSIZE, N_GAUSS) torch tensor
            - rs: (BSIZE) torch tensor
            - ds: (BSIZE) torch tensor
        """
        in_al = torch.cat([action, latent], dim=1)

        if self.mode == 'lstm':
            next_hidden = self.rnn(in_al, hidden)
        else:
            #print(f'hidden:{hidden}')
            if self.mode == 'dlf':
                h_0 = dict()
                h_1 = dict()
                for i in range(0, self.rnn.num_layers):
                    #print(f'h0:{hidden[0].shape}')
                    #print(f'h1:{hidden[1].shape}')
                    h0 = hidden[0].unsqueeze(1)
                    h1 = hidden[1].unsqueeze(1)
                    size = self.dlf_sys_order*self.dlf_num_head
                    #print(f'h0:{h0.shape} h1:{h1.shape} size:{size}')
                    h_0['layer_' + str(i)] = torch.nn.functional.interpolate(h0, size, mode='linear')
                    h_1['layer_' + str(i)] = torch.nn.functional.interpolate(h1, size, mode='linear')
                next_hidden0 = self.rnn(in_al, h_0)
                next_hidden1 = self.rnn(in_al, h_1)
                next_hidden = [next_hidden0, next_hidden1]
            else:
                next_hidden0 = self.rnn(in_al, hidden[0])
                next_hidden1 = self.rnn(in_al, hidden[1])
                next_hidden = [next_hidden0, next_hidden1]

        out_rnn = next_hidden[0]

        out_full = self.gmm_linear(out_rnn)

        stride = self.gaussians * self.latents

        mus = out_full[:, :stride]
        mus = mus.view(-1, self.gaussians, self.latents)

        sigmas = out_full[:, stride:2 * stride]
        sigmas = sigmas.view(-1, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = out_full[:, 2 * stride:2 * stride + self.gaussians]
        pi = pi.view(-1, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        r = out_full[:, -2]

        d = out_full[:, -1]

        return mus, sigmas, logpi, r, d, next_hidden
