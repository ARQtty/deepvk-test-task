import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from sklearn.metrics import accuracy_score, roc_auc_score



class CPCModel_NCE(nn.Module):
    '''
    Contrastive Predictive Coding model from https://arxiv.org/pdf/1807.03748.pdf.
    Consists of 5-layer convolution encoder that takes raw audio size Bx1x20480
    and produces z_t size Bx128xFeatures. Passes it to autoregressive RNN cell
    that produces c_t representation of audio with same 128 timesteps (Bx128xF)
    '''
    def __init__(self, config):
        super(CPCModel_NCE, self).__init__()
        self.device = config.train.device
        self.config = config
        # Fixed encoder parameters
        strides = [5, 4, 2, 2, 2]
        kernels = [10,8, 4, 4, 4]
        padding = [2, 2, 2, 2, 1]

        self.predict_steps = config.model.predict_steps
        self.context_size = config.model.context_size
        channels = config.model.conv_channels
        self.neg_samples = config.train.neg_samples

        # Creating encoder
        self.convolutions = []
        for i in range(5):
            dim = config.model.conv_channels
            if i == 0:
                dim = 1
            self.convolutions.append(nn.Conv1d(dim,
                                               channels,
                                               kernels[i],
                                               strides[i],
                                               padding[i]))
            self.convolutions.append(nn.ReLU())
            self.convolutions.append(nn.BatchNorm1d(channels))
        self.convolutions = nn.Sequential(*self.convolutions)

        # Creating autoregressor
        self.autoregressor = nn.GRU(channels,
                                    config.model.context_size,
                                    batch_first=True)

        # Transform that estimates c_t to z_t+k, unique for every k
        self.coupling_transforms = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(
                    config.model.context_size, channels)
            )
            for steps in range(self.predict_steps)
        ])

        self.loss = nn.LogSoftmax(dim=1)


    def _shift_z_rows(self, z):
        # Cuts first row and concatenate in to the end. So it is row shifting.
        # Works faster than randoming indexes

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
        # creates copy of z and smartly shuffling it to get tensor of
        # negative samples for every representation in z. Takes Theta(Z)
        # memory as z_neg is copy of z

        # z   b x l x c

        z = self._shift_z_rows(z)

        z = z.detach()
        # permutate columns in each row randomly
        z = z[:, torch.randperm(z.size(1)), :]

        # attach to grad tape again
        z.requires_grad_(True)
        return z


    def forward(self, x):
        # Takes B x 1 x L, L is fixed 20480

        batch_size = x.size()[0]
        # Encode
        for conv in self.convolutions:
            x = conv(x)

        # latent representation of audio fragment
        # z is B x C x L
        z = x

        # negatives for all the outputs z in batch. See _get_z_neg()
        z_negative = self._get_z_neg(z.permute(0,2,1))

        # autoregressor output
        ct, state = self.autoregressor(z.permute(0, 2, 1)) # b 128 256  and  b 1 256
        z = z.permute(0, 2, 1) # b x l x c

        logits = []
        labels = []
        losses = []
        # apply transforms for every predicted step k
        for k, transform in enumerate(self.coupling_transforms, start=1):
            # select z_k+t for a whole t
            z_pos = z[:, k:, :]
            z_neg = z_negative[:, k:, :]
            c     = ct[:, k:, :]

            # create negatives for every step
            if self.neg_samples >= x.size(0):
                # since the negative samples made by shifting audio representations over the batch axis (and
                # randomly permutated over columns (_get_z_neg())), round shifting by batch axis more then
                # %batch_size% times leads to repeating of negative samples, and i-th audio repr will become
                # a source of negative samples to i-th audio.
                # The paper allows negative sampling from current audio, but it will be disturb the convergence
                # so im warning about it here if it happens
                print('[WARN] positive samples occured in NCE loss')

            # create n negative samples for every timestep over the batch
            z_neg_full = z_neg.unsqueeze(1) # b x l x 1 x c
            for i in range(1, self.neg_samples):
                z_neg = self._shift_z_rows(z_neg)
                z_neg_full = torch.cat((z_neg_full, z_neg.unsqueeze(1)), dim=1)

            # estimate probabilities of z with filter (unique for each step, as mentioned in paper)
            estimated_c = transform(c).unsqueeze(1)    # b x 1 x l x c
            b, _, l, c = estimated_c.size()
            estimated_c = estimated_c.expand(b, self.config.train.neg_samples+1, l, c) # b x n+1 x l x c

            z_full = torch.cat((z_pos.unsqueeze(1), z_neg_full), dim=1) # b x n+1 x l x c

            # flatten
            b, s, l, c = z_full.size()
            z_full = z_full.reshape(b*l*s, c)

            b, s, l, c = estimated_c.size()
            estimated_c = estimated_c.reshape(b*s*l, c)
            # tensor magic of reducing dimensions
            z_full = z_full.unsqueeze(2)
            estimated = estimated_c.unsqueeze(1)
            f_k = torch.matmul(estimated, z_full).squeeze(1) # b*l*(n+1) x 1

            # forget about softmax *
            f_k = f_k.reshape(b*l, self.neg_samples+1)
            loss = self.loss(f_k)




            f_k = f_k.reshape(b*l*(self.neg_samples+1), 1)


            # create labels vector and flatting it. Neccessary for accuracy metric only
            answers_pos = torch.ones(b, 1, l, 1)
            answers_neg = torch.zeros_like(answers_pos).expand(b, self.neg_samples, l, 1)
            answers = torch.cat((answers_pos, answers_neg), dim=1)
            answers = answers.reshape(b*(self.neg_samples+1)*l, 1) # b*l*(n+1) x 1

            losses.append(-loss.sum() / ((128-k) * b))
            logits.append(f_k)
            labels.append(answers)

        return losses, logits, labels


    def predict(self, x):
        batch_size = x.size()[0]
        for conv in self.convolutions:
            x = conv(x)

        z = x.permute(0, 2, 1)
        ctx, state = self.autoregressor(z)
        return ctx, state


    @staticmethod
    def get_accuracy(logits, labels):
        to_numpy = lambda t : t.detach().cpu().squeeze().numpy()
        with torch.no_grad():
            labls = (to_numpy(logits) > 0.5).astype(int)
            accuracy = accuracy_score(to_numpy(labels), labls)
            return accuracy


    @staticmethod
    def get_auc_score(logits, labels):
        # useless, to be honest
        to_numpy = lambda t : t.detach().cpu().squeeze().numpy()
        with torch.no_grad():
            auc = roc_auc_score(to_numpy(labels), to_numpy(logits))
            return auc


    def get_summary(self, sample):
        # @param sample     tensor size (20480,)
        sample = sample.unsqueeze(0).unsqueeze(0)
        # it yields to jupyter cell automatically without returning it as value
        summary(self, sample.to(self.device))
