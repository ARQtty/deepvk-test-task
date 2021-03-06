import torch
import torch.nn as nn
import torch.nn.functional as F



class SpeakerClassificationModel(nn.Module):
    def __init__(self, cpc_model, hidden_size, n_speakers, freeze_cpc=True):
        super(SpeakerClassificationModel, self).__init__()
        self.cpc_model = cpc_model
        self.cpc_model.requires_grad_(not freeze_cpc)
        if freeze_cpc:
            self.cpc_model.eval()
        else:
            self.cpc_model.train()
        self.linear = nn.Sequential(nn.Linear(self.cpc_model.context_size, n_speakers))


    def load_cpc_checkpoint(self, checkpoint_path):
        print('Loading CPC weights')
        device = self.cpc_model.device
        self.cpc_model.load_state_dict(torch.load(checkpoint_path, map_location=device))


    def forward(self, x):
        _, ct_state = self.cpc_model.predict(x)
        ct_state = ct_state.squeeze()
        pred = self.linear(ct_state.squeeze(0))

        return pred


    @staticmethod
    def get_acc(logits, labels):
        with torch.no_grad():
            labels = F.one_hot(labels, num_classes=logits.size(1))
            matches = (logits.argmax(dim=1) == labels.argmax(dim=1)).int().sum()
            acc = matches.float() / logits.size(0)

            return acc
