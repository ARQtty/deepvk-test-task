import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from sklearn.metrics import accuracy_score, roc_auc_score

from .gim_block import GIMModule, GradBlock

strides = [5, 4, 2, 2, 2]
kernels = [10,8, 4, 4, 4]
padding = [2, 2, 2, 2, 1]



class GIMModel(nn.Module):
    def __init__(self):
        super(GIMModel, self).__init__()
        
        self.gim_modules = nn.ModuleList()
        for i in range(5):
            dim = 512
            if i == 0: dim = 1

            module = GIMModule(dim, 512, kernels[i], strides[i], padding[i], 2)
            if i%2 == 0:
                module.freeze()
            self.gim_modules.append(module)
     
    
    def freeze_block(self, block_ix):
        assert block_ix < len(self.gim_modules)
        self.gim_modules[block_ix].freeze()
        
        
    def unfreeze_block(self, block_ix):
        assert block_ix < len(self.gim_modules)
        self.gim_modules[block_ix].unfreeze()
        
            
    def forward(self, x):
#         print('input size', x.size())
        
        z = x
        for module in self.gim_modules:
            z = module(z)
#             print('feeding %s to next module' % str(z.size()))
    
#         print('final z size', z.size())
        return z
    
    
    def get_summary(self, sample):
        # sample is 1d tensor
        sample = sample.unsqueeze(0).unsqueeze(0)
        # it yields to jupyter cell automatically without returning it as value
        summary(self, sample.to(self.device))
