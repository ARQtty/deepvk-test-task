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
    def __init__(self, config):
        super(GILModel, self).__init__()
        self.config = config

        self.gim_modules = nn.ModuleList()
        for i in range(5):
            dim = 512
            if i == 0: dim = 1

            module = GILModule(dim, 512, kernels[i], strides[i], padding[i], 2)
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
            z = module(z)

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


    def get_summary(self, sample):
        # sample is 1d tensor
        sample = sample.unsqueeze(0).unsqueeze(0)
        # it yields to jupyter cell automatically without returning it as value
        summary(self, sample.to(self.device))