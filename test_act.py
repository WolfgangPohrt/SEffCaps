import torch
import time
import os
from pathlib import Path
import pickle
import sys
import librosa
import numpy as np
import yaml
from tqdm import tqdm
from torch.nn.functional import pad
from tools.beam import beam_decode, beam_decode_clustered
from tools.file_io import load_pickle_file
from models.TransModel import ACT
from dotmap import DotMap

# from tools.utils import decode_output
from re import sub
"""
Functions to test ACT with audio files 
in input directory.
"""
def _sentence_process(sentence, add_specials=True):

    # transform to lower case
    sentence = sentence.lower()

    if add_specials:
        sentence = '<sos> {} <eos>'.format(sentence)

    # remove any forgotten space before punctuation and double space
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')

    # remove punctuations
    sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')
    return sentence


def load_metadata(csv_file):
    """Load meta data of AudioCaps
    """
    if 'train' not in csv_file:
        caption_field = ['caption_{}'.format(i) for i in range(1, 6)]
    else:
        caption_field = None
    csv_list = load_csv_file(csv_file)

    audio_names = []
    captions = []

    for i, item in enumerate(csv_list):

        audio_name = item['file_name']
        if caption_field is not None:
            item_captions = [_sentence_process(item[cap_ind], add_specials=False) for cap_ind in caption_field]
        else:
            item_captions = _sentence_process(item['caption'])
        audio_names.append(audio_name)
        captions.append(item_captions)

    meta_dict = {'audio_name': np.array(audio_names), 'caption': np.array(captions)}

    return meta_dict

def write_pickle_file(obj, file_name):

    Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Write to {file_name} successfully.')





def extract_features(wav_path):
    sr = 32000
    window_length =  1024
    hop_length =  640 
    n_mels =  64
    waveform, _ = librosa.load(wav_path, sr=sr)
    feature = librosa.feature.melspectrogram(waveform, sr=sr, n_fft=window_length,
                                                 hop_length=hop_length, n_mels=n_mels)
    feature = librosa.power_to_db(feature).T
    feature = feature[:-1, :]
    return feature

def create_batch(wav_paths):
    batch_list = []
    for wav_path in wav_paths:
        feature = extract_features(wav_path=wav_path)   
        feature = torch.from_numpy(feature)
        feature = torch.unsqueeze(feature,0)
        batch_list.append(feature)
    print(batch_list)
    print([t.shape[1] for t in batch_list])
    max_size = max([t.shape[1] for t in batch_list])
    max_size = max_size if max_size == 500 else 500
    batch_list =[pad(t, (0, 0, 0 , max_size-t.shape[1]), 'constant', 0) for t in batch_list]

    return torch.cat(batch_list, 0)

def get_config():
    with open('settings/settings.yaml', 'r') as f:

        config = yaml.load(f, Loader=yaml.FullLoader)
        config = DotMap(config)
        return config


def get_act_model(device):
    config = get_config()
    words_list = load_pickle_file('pickles/words_list.p')
    ntokens = len(words_list)
    model = ACT(config, ntokens)
    model.load_state_dict(torch.load('/home/theokouz/src/ACT/pretrained_models/ACTm.pth')['model'])
    model.to(device)
    model.eval()
    return model, words_list


def forward_pass(batch, words_list, model, device):
    batch = batch.to(device)
    sos_ind = words_list.index('<sos>')
    eos_ind = words_list.index('<eos>')
    with torch.no_grad():
        # output = beam_decode(batch, model, sos_ind, eos_ind, beam_width=2)
        output = beam_decode_clustered(batch, model, sos_ind, eos_ind, beam_width=2)
        print('beam finished')
        output = output[:, 1:].int()
        y_hat_batch = torch.zeros(output.shape).fill_(eos_ind).to(device)
        y_hat_batch = y_hat_batch.int()
        for i in range(output.shape[0]):  # batch_size
            for j in range(output.shape[1]):
                y_hat_batch[i, j] = output[i, j]
                if output[i, j] == eos_ind:
                    break
                elif j == output.shape[1] - 1:
                    y_hat_batch[i, j] = eos_ind
        return y_hat_batch
