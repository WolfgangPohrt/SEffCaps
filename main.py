from asyncio import subprocess
import os
from statistics import mode
import sys
import csv
import argparse
import numpy as np
import torch
from tools.post_proc import remove_speaking
import torchaudio
from glob import glob
from test_act import create_batch, get_act_model, forward_pass
from tools.add_subs import add_subs, create_subs_srt
from ast_model import ASTModel
from tools.ast_tools import load_label, make_features, get_ast_model, classify_wav
from tools.file_io import get_config, load_pickle_file
from tools.video_audio_utils import download_video, extract_audio, segment_audio
from tools.dataloader import get_data_loader, get_data_loader_passt
from passt_model.caption_passt import captionPaSST



if __name__ == '__main__':
    torchaudio.set_audio_backend("soundfile") 
    parser = argparse.ArgumentParser(description='Example of parser:'
                                                 'python inference --audiodir_path ./test_dir '
                                                 '--model_path ./pretrained_models/audioset_10_10_0.4593.pth')


    parser.add_argument('--audiodir_path',
                        help='the to save the audio',
                        type=str, required=True)
    parser.add_argument('--url',
                        help='youtube url',
                        type=str, required=True) 

    config = get_config('settings/settings.yaml')
    use_passt = False
    segments_duration = 10
    # device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    workingdir=args.audiodir_path
    label_csv = config.ast.audioset_labels
    basename = 'test'
    ignore_tags = ['Speech', 'Music']
    video_path = download_video(url=args.url, workingdir=workingdir, basename=basename)
    video_path = 'downloaded_audio/test.mp4'
    audio_path = extract_audio(video_path=video_path)
    audio_paths = segment_audio(audio_path=audio_path, seg_dur=segments_duration)

    labels_out = []
    # Load preatrained AST for taging
    input_tdim = 1024
    audio_model = get_ast_model(input_tdim=input_tdim, device=device, config=config)
    
    for i, audio_path in enumerate(audio_paths):
        label = classify_wav(audio_path, audio_model, label_csv)
        labels_out.append(label)
    print(labels_out)
    
    # find video segments with tags not in the ignore list
    caption_segs = []
    for i, label in enumerate(labels_out):
        if label not in ignore_tags:
            caption_segs.append([str(i).zfill(3), label, i*segments_duration, segments_duration*(i+1)])
    print(caption_segs)  
    # audio caption from the segments
    paths_to_caption = [f'{workingdir}/{basename}{i[0]}.wav' for i in caption_segs]
    batch = create_batch(paths_to_caption)
    # print(batch.shape)  
    print(paths_to_caption)
    

    if use_passt:
        config_passt = get_config('settings/settings_passt.yaml')
        words_list = load_pickle_file(config_passt.path.vocabulary)
        ntokens = len(words_list)
        model = captionPaSST(config_passt, ntokens)
        model.load_state_dict(torch.load('/home/theokouz/src/ACT/outputs/passt_s_swa_p16_s16_128_ap473_w2v/model/best_model.pth')['model'])
        dataloader = get_data_loader_passt(paths_to_caption)
    else:
        model, words_list = get_act_model(device)
        dataloader = get_data_loader(paths_to_caption)
    sos_ind = words_list.index('<sos>')
    eos_ind = words_list.index('<eos>')
    # y_hat = forward_pass(batch=batch, words_list=words_list, model=model, device=device)
    # total_captions = [[words_list[idx] for idx in pred if idx != eos_ind] for pred in y_hat]
    total_captions = []
    for batch in dataloader:
        y_hat = forward_pass(batch=batch, words_list=words_list, model=model, device=device)
        caption_pred = [[words_list[idx] for idx in pred if idx != eos_ind] for pred in y_hat]
        total_captions+=caption_pred

    timestamps = [[' '.join(p)] + s[-2:] for p, s in zip(total_captions, caption_segs)]
    print(timestamps)
    caps_timestamps_clean = remove_speaking(timestamps)
    print(caps_timestamps_clean)
    subs_path  = 'test_subs.srt'
    input_path = video_path
    output_path = 'test_video_subs.mp4'
    create_subs_srt(subs_path, caps_timestamps_clean)
    # add_subs(input_path, subs_path, output_path)