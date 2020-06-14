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



class GILModule(nn.Module):
    def __init__(self, conv_dim_in, conv_dim_out, kernel, stride, padding, config):
        super(GILModule, self).__init__()

        self.gradient_modifier = EmptyModifier # unfreezed
        self.cpc_module = CPCModule(conv_dim_in, conv_dim_out, kernel, stride, padding, config)
        self.grad_block = GradBlock()


    def freeze(self):
        self.gradient_modifier = no_grad
        self.cpc_module.set_grad_calc(False)

    def unfreeze(self):
        self.gradient_modifier = EmptyModifier
        self.cpc_module.set_grad_calc(True)

    def is_freezed(self):
        return self.cpc_module.is_freezed()


    def forward(self, x):
        # if not self.training: #while testing
        #     modifier = no_grad
        # else:
        #     modifier = self.gradient_modifier
        with self.gradient_modifier():
            z, ct, ct_state, losses = self.cpc_module(x)
            ct = self.grad_block(ct)

        return z, ct, ct_state, losses


    def predict(self, z):
        z, ct, state_ct = self.cpc_module.predict(z)
        return z, ct, state_ct
