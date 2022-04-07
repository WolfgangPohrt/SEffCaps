import torch
from passt import get_model
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import yaml
from tools.utils import align_word_embedding
from dotmap import DotMap
from tqdm import tqdm

def get_config():

    with open('settings.yaml', 'r') as f:

        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)
    return config




class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class captionPaSST(nn.Module):

    def __init__(self, config, ntoken):
        super(captionPaSST, self).__init__()

        self.ntoken = ntoken
        self.model_name = config.encoder.model
        self.fstride = config.encoder.fstride
        self.tstride = config.encdoer.tstride
        self.u_patchout = config.encoder.u_patchout
        self.s_patchout_t = config.encdoer.s_patchout_t 
        self.s_patchout_f = config.encdoer.s_patchout_f 

        self.encoder = get_model(arch="passt_s_swa_p16_128_ap476", pretrained=True, n_classes=527, in_channels=1,
                   fstride=16, tstride=16,input_fdim=128, input_tdim=998,
                   u_patchout=0, s_patchout_t=40, s_patchout_f=4)
        # self.encoder = get_model(arch=config.encoder.model, pretrained=True, n_classes=527, in_channels=1,
        #            fstride=config.encoder.fstride, tstride=config.encdoer.tstride ,input_fdim=128, input_tdim=998,
        #            u_patchout=config.encoder.u_patchout, s_patchout_t=config.encdoer.s_patchout_t,
        #            s_patchout_f=config.encdoer.s_patchout_f)
        if config.encoder.freeze:
            for name, p in self.encoder.named_parameters():
                p.requires_grad = False

        # settings for decoder
        nhead = config.decoder.nhead
        nlayers = config.decoder.nlayers
        dim_feedforward = config.decoder.dim_feedforward
        activation = config.decoder.activation
        dropout = config.decoder.dropout
        self.nhid = config.decoder.nhid

        self.encoder_linear = nn.Linear(768, 512)

        self.pos_encoder = PositionalEncoding(self.nhid, dropout)

        decoder_layers = TransformerDecoderLayer(self.nhid,
                                                 nhead,
                                                 dim_feedforward,
                                                 dropout,
                                                 activation)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

        self.word_emb = nn.Embedding(ntoken, self.nhid)
        self.dec_fc = nn.Linear(self.nhid, ntoken)
        # setting for pretrained word embedding
        if config.word_embedding.freeze:
            self.word_emb.weight.requires_grad = False
        if config.word_embedding.pretrained:
            self.word_emb.weight.data = align_word_embedding(config.path.vocabulary,
                                                             config.path.word2vec,
                                                             self.nhid)
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, src):
        """
        Args:
            src: spectrogram, batch x time x n_mels
        """
        src, _ = self.encoder(src.to('cuda'))  # batch x time x 527
        src = F.relu_(self.encoder_linear(src))  # batch x time x nhid
        src = src.transpose(0, 1)  # time x batch x nhid
        return src

    def decode(self, encoded_feats, tgt, input_mask=None, target_mask=None, target_padding_mask=None):
        # tgt: (batch_size, caption_length)
        # encoded_feats: (T, batch_size, nhid)

        tgt = tgt.transpose(0, 1)
        if target_mask is None or target_mask.size()[0] != len(tgt):
            device = tgt.device
            target_mask = self.generate_square_subsequent_mask(len(tgt)).to(device)

        tgt = self.word_emb(tgt) * math.sqrt(self.nhid)
        tgt = self.pos_encoder(tgt)

        output = self.transformer_decoder(tgt, encoded_feats,
                                          memory_mask=input_mask,
                                          tgt_mask=target_mask,
                                          tgt_key_padding_mask=target_padding_mask)
        output = self.dec_fc(output)

        return output

    def forward(self, src, tgt, input_mask=None, target_mask=None, target_padding_mask=None):
        # src: spectrogram

        encoded_feats = self.encode(src)
        output = self.decode(encoded_feats, tgt,
                             input_mask=input_mask,
                             target_mask=target_mask,
                             target_padding_mask=target_padding_mask)
        return output

if __name__ == "__main__":
    config = get_config()
    src = torch.ones((8,1,128,1000))
    tgt = torch.ones((8,22))
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
    device = 'cpu'
    tgt = tgt.long()
    model = captionPaSST(config, 1312)
    # model = nn.DataParallel(model, device_ids = [2, 3])
    model.to(device)

    for _ in tqdm(range(30000)):
        out = model(src.to(device), tgt.to(device))
        print(out.shape)
        break
