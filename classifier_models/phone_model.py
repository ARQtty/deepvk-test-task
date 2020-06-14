import torch
import torch.nn as nn
import torch.nn.functional as F



class PhonesClassificationModel(nn.Module):
    def __init__(self, cpc_model, hidden_size, n_phones, freeze_cpc=True):
        super(PhonesClassificationModel, self).__init__()
        self.cpc_model = cpc_model
        self.cpc_model.requires_grad_(not freeze_cpc)
        if freeze_cpc:
            self.cpc_model.eval()
        else:
            self.cpc_model.train()
        
        self.linear = nn.Sequential(nn.Linear(self.cpc_model.context_size, n_phones))
        
        
    def load_cpc_checkpoint(self, checkpoint_path):
        print('Loading CPC weights')
        device = self.cpc_model.device
        self.cpc_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        
    def forward(self, x):
        ct, ct_state = self.cpc_model.predict(x)
        b, l, c = ct.size()
        ct = ct.reshape(b*l, c)
        pred = self.linear(ct)

        return pred


    @staticmethod
    def get_acc(logits, labels):
        with torch.no_grad():
            labels = F.one_hot(labels, num_classes=72)
            matches = (logits.argmax(dim=1) == labels.argmax(dim=1)).int().sum()
            acc = matches.float() / logits.size(0)

            return acc
