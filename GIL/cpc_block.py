import torch
import torch.nn as nn
import torch.nn.functional as F



def encoder_fabric(dim_in, dim_out, kernel, stride, padding):
    convolutions = []
    convolutions.append(nn.Conv1d(dim_in,
                                  dim_out,
                                   kernel,
                                   stride,
                                   padding))
    convolutions.append(nn.ReLU())
    convolutions.append(nn.BatchNorm1d(dim_out))
    return nn.Sequential(*convolutions)



class CPCModule(nn.Module):
    def __init__(self, conv_dim_in, conv_dim_out, kernel, stride, padding, config):
        super(CPCModule, self).__init__()
        self.calculate_grads = True
        self.config = config

        self.encoder = encoder_fabric(conv_dim_in, conv_dim_out, kernel, stride, padding)
        self.autoregressor = nn.GRU(config.model.conv_channels, config.model.context_size)

        self.transforms = nn.ModuleList([
                                        nn.Sequential(
                                            nn.Linear(conv_dim_out, config.model.conv_channels))
                                        for k in range(config.model.predict_steps)])
        self.predicting_steps = config.model.predict_steps
        self.loss = nn.LogSoftmax(dim=1)

        self.opt = torch.optim.Adam(self.parameters(), lr=config.train.lr)


    def set_grad_calc(self, val):
        assert isinstance(val, bool)
        self.calculate_grads = val

    def is_freezed(self):
        return not self.calculate_grads


    def _shift_z_rows(self, z):
        # z    b x l x c
        z = z.detach()

        # take last audio latent repr
        z_last = z[-1, :, :].unsqueeze(0)
        # and shift audio reprs along the 0th dim
        z = z[:-1]
        z = torch.cat((z_last, z), dim=0)

        z.requires_grad_(True)
        return z


    def _get_z_neg(self, z):
        # creates noise copy of z by smartly shuffling data
        # z   b x l x c

        z = self._shift_z_rows(z)

        z = z.detach()
        # permutate columns in each row randomly
        z = z[:, torch.randperm(z.size(1)), :]

        # attach to grad tape again
        z.requires_grad_(True)
        return z


    def forward(self, x):
        z = self.encoder(x)
        z = z.permute(0, 2, 1)
        #         print('  z size', z.size())
        z_negative = self._get_z_neg(z)

        ct, state = self.autoregressor(z)

        losses = []
        for k, transform in enumerate(self.transforms, start=1):
            #             print('  transform %d---'%k)
            z_pos = z[:, k:, :]
            z_neg = z_negative[:, k:, :]
            #print('    z pos' , z_pos.size(), 'z neg', z_neg.size())

            if self.config.train.neg_samples >= x.size(0):
                print('[WARN] positive samples occured in NCE loss')

            # create n negative samples for every timestep over the batch
            z_neg_full = z_neg.unsqueeze(1) # b x l x 1 x c
            for i in range(1, self.config.train.neg_samples):
                z_neg = self._shift_z_rows(z_neg)
                z_neg_full = torch.cat((z_neg_full, z_neg.unsqueeze(1)), dim=1)


            estimated_pos = transform(z_pos).squeeze(0).unsqueeze(1) # b x 1 x l x c
            estimated_neg = transform(z_neg_full).squeeze(0)         # b x n x l x c

            # concatinate pos and neg estimated samples
            estimated = torch.cat((estimated_pos, estimated_neg), dim=1) # b x n+1 x l x c
            b, s, l, c = estimated.size()
            estimated = estimated.reshape(b*s*l, c)

            z_full = torch.cat((z_pos.unsqueeze(1), z_neg_full), dim=1) # b x n+1 x l x c

            # flatten
            b, s, l, c = z_full.size()
            z_full = z_full.reshape(b*l*s, c)

            # tensor magic of reducing dimensions
            z_full = z_full.unsqueeze(2)
            estimated = estimated.unsqueeze(1)
            f_k = torch.matmul(estimated, z_full).squeeze(1) # b*l*(n+1) x 1

            # forget about softmax *
            f_k = f_k.reshape(b*l, self.config.train.neg_samples+1)
            #print('    fk size', f_k.size())

            loss = self.loss(f_k)
            losses.append(-loss.sum() / ((l - k)*b))

        if self.calculate_grads and self.training:
            for loss in losses:
                loss.backward(retain_graph=True)
                self.opt.step()
            self.opt.zero_grad()

        return z.permute(0, 2, 1)
