import torch
import torch.nn as nn
import torch.nn.functional as F

from .cpc_block import CPCModule
from .grad_modifiers import no_grad, EmptyModifier



class GradBlock(nn.Module):
    def __init__(self):
        super(GradBlock, self).__init__()
    
    def forward(self, z):
        z = z.detach()
        return z
    
    
    
class GIMModule(nn.Module):
    def __init__(self, conv_dim_in, conv_dim_out, kernel, stride, padding, predicting_steps):
        super(GIMModule, self).__init__()
        
        self.gradient_modifier = EmptyModifier
        self.cpc_module = CPCModule(conv_dim_in, conv_dim_out, kernel, stride, padding, predicting_steps)
        self.grad_block = GradBlock()
        
    
    def freeze(self):
        self.gradient_modifier = no_grad
        self.cpc_module.set_grad_calc(False)
        
    def unfreeze(self):
        self.gradient_modifier = EmptyModifier
        self.cpc_module.set_grad_calc(True)
        
    
    def forward(self, x):
        with self.gradient_modifier():
            z = self.cpc_module(x)
            z = self.grad_block(z)

        return z