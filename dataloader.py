import torch
import numpy as np
import librosa
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
# from test_act import extract_features
from torch.nn.functional import pad


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

class AudioData(Dataset):

    def __init__(self, audio_paths):
        super(AudioData, self).__init__()

        self.audio_paths = audio_paths 

    def __len__(self):
            return len(self.audio_paths)

    def __getitem__(self, index):
        
        audio_path = self.audio_paths[index]
        feature = extract_features(wav_path=audio_path)   
        feature = torch.from_numpy(feature)
        feature = torch.unsqueeze(feature,0)
        return feature

def collate_fn(batch):
    max_size = max([t.shape[1] for t in batch])
    max_size = max_size if max_size == 500 else 500
    batch_list =[pad(t, (0, 0, 0 , max_size-t.shape[1]), 'constant', 0) for t in batch]
    return torch.cat(batch_list, 0)


def get_data_loader(audio_paths):
    dataset = AudioData(audio_paths)
    data_loader = DataLoader(dataset=dataset, batch_size=4,
                            shuffle=False, drop_last=False,
                            num_workers=4, collate_fn=collate_fn)
    return data_loader

# if __name__ == '__main__':
#     audio_paths = ['downlaoded_audio/test000.wav', 'downlaoded_audio/test001.wav', 'downlaoded_audio/test002.wav', 'downlaoded_audio/test003.wav']
#     data_loader = get_data_loader(audio_paths)
#     for i, data in enumerate(data_loader):
#         print(data.shape)
