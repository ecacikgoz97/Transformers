import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, embedding_size=512, src_vocab_size, trg_vocab_size, src_pad_idx, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, forward_expansion=4, dropout=0.1, max_len=100, device):
        super(Transformer, self).__init__()
        self.device = device
        self.src_word_embedding = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=embedding_size)
        self.trg_word_embedding = nn.Embedding(num_embeddings=trg_vocab_size, embedding_dim=embedding_size)
        self.src_position_embedding = nn.Embedding(num_embeddings=max_len, embedding_dim=embedding_size)
        self.trg_position_embedding = nn.Embedding(num_embeddings=max_len, embedding_dim=embedding_size)
        self.transformer = nn.Transformer(d_model=embedding_size, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=forward_expansion, dropout=dropout)
        self.fc = nn.Linear(in_features=embedding_size, out_features=trg_vocab_size)
        self.dropout = nn.Dropout(p=dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N).to(self.device))
        trg_positions = (torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).to(self.device))

        embed_src = self.dropout((self.src_word_embedding(src) + self.src_position_embedding(src_positions)))
        embed_trg = self.dropout((self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)))

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        out = self.transformer(embed_src, embed_trg, src_key_padding_mask=src_padding_mask, tgt_mask=trg_mask)
        out = self.fc(out)
        return out
