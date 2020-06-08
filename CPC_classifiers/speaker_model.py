import torch
import torch.nn as nn
import torch.nn.functional as F



class SpeakerClassificationCPC(nn.Module):
    def __init__(self, cpc_model, hidden_size, n_speakers, freeze_cpc=True):
        super(SpeakerClassificationCPC, self).__init__()
        self.cpc_model = cpc_model
        self.cpc_model.requires_grad_(not freeze_cpc)
        
#         self.linear = nn.Sequential(nn.Linear(self.cpc_model.context_size, hidden_size),
#                                     nn.BatchNorm1d(hidden_size),
#                                     nn.Tanh(),
#                                     nn.Dropout(0.2),
#                                     nn.Linear(hidden_size, hidden_size//3),
#                                     nn.BatchNorm1d(hidden_size//3),
#                                     nn.Tanh(),
#                                     nn.Dropout(0.2),
#                                     nn.Linear(hidden_size//3, n_speakers)).requires_grad_(True)
        
        self.linear = nn.Sequential(nn.Linear(self.cpc_model.context_size, n_speakers))    
        
        
    def load_cpc_checkpoint(self, checkpoint_path):
        device = self.cpc_model.device
        self.cpc_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        
    def forward(self, x):
        _, ct = self.cpc_model.predict(x)
        ct = ct.squeeze(1)
        pred = self.linear(ct.squeeze(0))
        
        return pred
    
    
    @staticmethod
    def get_acc(logits, labels):
        with torch.no_grad():
            labels = F.one_hot(labels, num_classes=logits.size(1))
            matches = (logits.argmax(dim=1) == labels.argmax(dim=1)).int().sum()
            acc = matches.float() / logits.size(0)

            return acc        