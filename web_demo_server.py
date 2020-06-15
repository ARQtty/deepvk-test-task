from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# ============================================================
# decoding binary encoded data to tensor with samplingrate 16k
import base64
import pickle
import subprocess
import torchaudio
from torchaudio.transforms import Resample
import torch
from hparams import Hparam
from CPC_model.model import CPCModel_NCE
from classifier_models.speaker_model import SpeakerClassificationModel


def decode_base64(encoded):
    return base64.urlsafe_b64decode(encoded[21:])

def create_tmp_file(binary_obj):
    with open('templates/tmp.webm', 'wb') as f:
        f.write(binary_obj)

def convert_tmp_file():
    cmd = 'ffmpeg -y -i templates/tmp.webm -vn templates/tmp.wav'
    subprocess.call(cmd.split())

def load_as_tensor(transform):
    wf, sampling_rate = torchaudio.load('templates/tmp.wav')
    wf = transform(wf)
    return wf

resample = Resample(48000, 16000)

def pipeline(encoded):
    create_tmp_file(decode_base64(encoded))
    convert_tmp_file()
    return load_as_tensor(resample)

# ============================================================
class Model:
    def __init__(self, classifier_config_path):
        clf_cfg = Hparam(classifier_config_path)
        cpc_cfg = Hparam(clf_cfg.model.cpc_config_path)
        self.device = clf_cfg.train.device

        speakers_bank = pickle.load(open('templates/mean_speakers_vecs_dict.pkl', 'rb'))
        self.speakers, self.mean_vecs = list(speakers_bank.keys()), torch.stack(list(speakers_bank.values()), dim=0)

        model_cpc = CPCModel_NCE(cpc_cfg).to(clf_cfg.train.device)
        self.model = SpeakerClassificationModel(model_cpc,
                                           clf_cfg.model.hidden_size,
                                           40,
                                           clf_cfg).to(clf_cfg.train.device)
        self.model.load_state_dict(torch.load(clf_cfg.train.checkpoints_dir + '/' + clf_cfg.train.cpc_checkpoint))
        self.model.eval()


    def preprocess_tensor(self, tensor):
        assert len(tensor.size()) == 2
        length = tensor.size(1)
        parts = length // 20480
        if length % 20480 > 0.3 * 16000: # greater than 0.3sec
            new_t = torch.zeros((1, 20480*(parts+1)))
            new_t[0, :length] = tensor
            tensor = new_t
        else: # or throw it away
            tensor = tensor[0, :length]

        tensor = tensor.view(-1, 20480)
        return tensor


    def get_scores(self, audio_batch):
        with torch.no_grad():
            preds = self.model(audio_batch.unsqueeze(1).to(self.device))
            return preds


    def get_labels_cosine(self, scores_batch):
        # by cosine dist
        tn1 = scores_batch / scores_batch.norm(dim=1).unsqueeze(1)
        tn2 = self.mean_vecs / self.mean_vecs.norm(dim=1)
        cosins = torch.mm(tn1, tn2)
        nearest_ixs = cosins.argmax(dim=1)
        return nearest_ixs


    def get_labels_eucl(self, scores_batch):
        # by eucleadian dist, worse
        scores = torch.mm(scores_batch, self.mean_vecs)
        nearest_ixs = scores.argmax(dim=1)
        return nearest_ixs


    def get_labels_max(self, scores_batch):
        return scores_batch.argmax(dim=1)

# ============================================================




@app.route("/", methods=['GET'])
def handle():
    return render_template('index.html')

@app.route("/audio", methods=['POST'])
def handle1():
    audiocontent = request.form['k']
    inp = pipeline(audiocontent)
    print('got tensor size', inp.size())
    inp = model.preprocess_tensor(inp)
    scores = model.get_scores(inp)
    labels = model.get_labels_cosine(scores)

    return jsonify({'status': 'ok', 'labels': labels.tolist()});


if __name__ == "__main__":
    model = Model('templates/config_classifier.yaml')
    app.run(debug=False, host='127.0.0.1', port=5022)
