import torch
import torchaudio
import csv
import numpy as np

from ast_model import ASTModel


def find_segments_to_caption(labels, segments_duration, ignore_tags):
    caption_segs = []
    for i, labs in enumerate(labels):
        if labs[0][0] not in ignore_tags:
            label = labs[0][0]
            caption_segs.append([str(i).zfill(3), label, i*segments_duration, segments_duration*(i+1)])
        else:
            if labs[0][1] < 0.2:
                label = labs[1][0]
                caption_segs.append([str(i).zfill(3), label, i*segments_duration, segments_duration*(i+1)])
            
        
    return caption_segs
        



def classify_wav(audio_path, audio_model, label_csv):
    waveform, sr = torchaudio.load(audio_path)
    # 1. make feature for predict
    feats = make_features(audio_path, mel_bins=128)           # shape(1024, 128)
    input_tdim = feats.shape[0]
    # 2. feed the data feature to model
    feats_data = feats.expand(1, input_tdim, 128)           # reshape the feature
    audio_model.eval()                                      # set the eval model
    with torch.no_grad():
        output = audio_model.forward(feats_data)
        output = torch.sigmoid(output)
    result_output = output.data.cpu().numpy()[0]
    # 4. map the post-prob to label
    labels = load_label(label_csv)
    sorted_indexes = np.argsort(result_output)[::-1]
    print(result_output[sorted_indexes[0]], np.array(labels)[sorted_indexes[0]], result_output[sorted_indexes[1]], np.array(labels)[sorted_indexes[1]])

    return [(np.array(labels)[sorted_indexes[i]], result_output[sorted_indexes[i]]) \
                for i in range(3)]

def get_ast_model(input_tdim, device, config):
    ast_mdl = ASTModel(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=True, audioset_pretrain=True)
    checkpoint_path = config.ast.pretrained_model
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    audio_model = audio_model.to(device)
    return audio_model

def make_features(wav_name, mel_bins, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


def load_label(label_csv):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels