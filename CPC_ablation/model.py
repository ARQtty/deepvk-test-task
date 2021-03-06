import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary



class CPCModel(nn.Module):
    '''
    CPC model without neg sampling and with different (compare to orig) loss
    ref my pure implementation in CPC_model for detailed comments
    '''
    def __init__(self, config):
        super(CPCModel, self).__init__()
        self.device = config.train.device
        strides = [5, 4, 2, 2, 2]
        kernels = [10,8, 4, 4, 4]
        padding = [2, 2, 2, 2, 1]

        self.predict_steps = config.model.predict_steps
        self.context_size = config.model.context_size
        channels = config.model.conv_channels

        # Encoder
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

        # Autoregression
        self.autoregressor = nn.GRU(channels,
                                    config.model.context_size,
                                    batch_first=True)

        # Transforms from c_t -> z_t+k
        self.coupling_transforms = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    channels, channels, kernel_size=1)
            )
            for steps in range(self.predict_steps)
        ])


    def forward(self, x):
        batch_size = x.size()[0]
        for conv in self.convolutions:
            x = conv(x)

        z = x.permute(0, 2, 1)
        ctx, state = self.autoregressor(z)
        z = z.permute(0, 2, 1)


        logits = []
        labels = []
        for i in range(len(self.coupling_transforms)):
            estimated = self.coupling_transforms[i](z)
            logit = torch.bmm(z.permute(0, 2, 1), estimated) # b x f x f

            label = torch.eye(logit.size(2) - (i+1)).to(self.device)
            label = F.pad(label, (0, i+1, i+1, 0))
            label = label.unsqueeze(0).expand_as(logit)

            logits.append(logit)
            labels.append(label)

        return logits, labels


    def predict(self, x):
        batch_size = x.size()[0]
        for conv in self.convolutions:
            x = conv(x)

        z = x.permute(0, 2, 1)
        ctx, state = self.autoregressor(z)
        return ctx, state


    @staticmethod
    def get_accuracy(logits, labels):
        with torch.no_grad():
            logits[logits.ge(0.5)] = 1
            logits[logits < 0.5] = 0
            acc = (logits == labels).sum().float() / logits.numel()
            return acc


    def get_summary(self, sample):
        # sample is 1d tensor
        sample = sample.unsqueeze(0).unsqueeze(0)
        # it yields to jupyter cell automatically without returning it as value
        summary(self, sample.to(self.device))
