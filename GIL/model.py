import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from sklearn.metrics import accuracy_score, roc_auc_score

from .gil_block import GILModule, GradBlock


strides = [5, 4, 2, 2, 2]
kernels = [10,8, 4, 4, 4]
padding = [2, 2, 2, 2, 1]



class GILModel(nn.Module):
    def __init__(self, config, writer):
        super(GILModel, self).__init__()
        self.config = config
        self.device = config.train.device
        self.context_size = config.model.context_size
        assert config.train.n_blocks == len(strides) # stupid way to check if we forget to change it

        self.writer = writer
        self.train_upd_step = 0
        self.test_upd_step = 0

        self.gim_modules = nn.ModuleList()
        for i in range(config.train.n_blocks):
            dim = config.model.conv_channels
            if i == 0: dim = 1

            module = GILModule(dim,
                               config.model.conv_channels,
                               kernels[i],
                               strides[i],
                               padding[i],
                               config)
            self.gim_modules.append(module)


    def freeze_block(self, block_ix):
        assert block_ix < len(self.gim_modules)
        self.gim_modules[block_ix].freeze()


    def unfreeze_block(self, block_ix):
        assert block_ix < len(self.gim_modules)
        self.gim_modules[block_ix].unfreeze()


    def forward(self, x):
        z = x
        for i, module in enumerate(self.gim_modules):
            z, losses = module(z)

            # write train logs
            if self.training:
                train_test = 'Train'
                step = self.train_upd_step
                self.train_upd_step += 1
            else:
                train_test = 'Test'
                step = self.test_upd_step
                self.test_upd_step += 1
            avg_loss = sum(losses) / max(1, len(losses))
            self.writer.add_scalar('%s_loss/module_%d_loss' % (train_test, i), avg_loss, step)

            # check if all next modules are freezed, flag "skip successors"
            # in config is true and we are in training, we can skip
            # successors calculations and lost nothing
            all_freezed = True
            for successor in self.gim_modules[i+1:]:
                if not successor.is_freezed():
                    all_freezed = False
            if all_freezed and self.config.train.unfreezing.skip_freezed_successors:
                return z

        return z


    def predict(self, x):
        batch_size = x.size()[0]
        z = x
        with torch.no_grad():
            for module in self.gim_modules:
                z, ct, state_ct = module.predict(z)

        return ct, state_ct


    def get_summary(self, sample):
        # sample is 1d tensor
        sample = sample.unsqueeze(0).unsqueeze(0)
        # it yields to jupyter cell automatically without returning it as value
        summary(self, sample.to(self.device))
