"""
Action Recognition Models
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from configs import load_model, load_embedding_fn

from src.models.s4.s4d import S4D

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d


class MyS4(nn.Module):
    def __init__(self, model_params, device=None):

        super().__init__()

        l_rate = model_params['lr']
        d_input = model_params['d_input']
        d_output = model_params['d_output']
        n_layers = model_params['n_layers']
        d_model = model_params['d_model']
        dropout = model_params['dropout']
        prenorm = model_params['prenorm']

        #self.model = S4D()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout,
                    transposed=True, lr=min(0.001, l_rate))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        hidden_dim = self.get_hidden_size()

        #self.decoder = nn.Linear(d_model, d_output)

        self.decoder = nn.Sequential(nn.Linear(hidden_dim, d_model),
                                     nn.Dropout(0.5),
                                     nn.Tanh(),
                                     nn.Linear(d_model, d_output))

    def forward(self, x, dummy_lens):
        #print('s4 works!', self.model)

        # print(x.shape)
        # print(x.dtype)

        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        # print(x.shape)
        # print(x.dtype)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        # print(x.shape)
        # print(x.dtype)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        logits = self.decoder(x)  # (B, d_model) -> (B, d_output)

        # print(logits.shape)

        return logits


class BiLSTM(nn.Module):
    def __init__(self, model_params, device=None):
        super().__init__()

        input_dim = model_params['input_dim']
        hidden_dim = model_params['lstm_hidden']
        num_layers = model_params['num_layers']
        lstm_dropout = model_params['lstm_dropout']
        fc_dim = model_params['fc_dim']
        num_cls = model_params['num_classes']

        # BiLSTM
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                              dropout=lstm_dropout, bidirectional=True)

        # Logit layer
        self.fc = nn.Sequential(nn.Linear(2 * hidden_dim, fc_dim),
                                nn.Dropout(0.5),
                                nn.Tanh(),
                                nn.Linear(fc_dim, num_cls))
        # Params
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x_input, x_seq_len):
        # x_input: [batch_size, seq_len, input_dim]

        x = pack_padded_sequence(
            x_input, x_seq_len, batch_first=True, enforce_sorted=False)

        # outputs: [sum_{i=0}^batch (seq_lens[i]), 2 * hidden_dim]
        outputs, (hidden, cell) = self.bilstm(x)

        # hidden: [num_layers * 2, batch_size, hidden_dim]
        # [num_layers, 2, batch_size, hidden_dim]
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)

        # Skip hidden states of intermediate layers
        # [2, batch_size, hidden_dim]
        hidden = hidden[-1]

        # Extract the forward & backward hidden states
        forward_h = hidden[0]
        backward_h = hidden[1]

        # Concatenate hidden states
        # [batch_size, 2*hidden_dim]
        final_hidden = torch.cat([forward_h, backward_h], dim=1)

        # [batch_size, num_cls]
        logits = self.fc(final_hidden)

        return logits


# ---------------------------------------------------------------
class Transformer(nn.Module):
    """
    Adapts HuggingFace's Transformer for handling video embeddings
    """

    def __init__(self, model_params, device=None):
        super(Transformer, self).__init__()

        input_dim = model_params['input_dim']
        # e.g. bert, roberta, etc.
        model_name = model_params['model_name']
        # e.g. bert-base-uncased, roberta-base, etc.
        config_name = model_params['config_name']
        config_dict = model_params['config_dict']       # custom config params
        use_pretrained = model_params['use_pretrained']
        fc_dim = model_params['fc_dim']
        num_cls = model_params['num_classes']

        self.max_len = model_params['max_video_len']

        self.device = device

        # Load transformer for the given name & config
        self.transformer = load_model(
            model_name, config_dict, config_name, use_pretrained)

        hidden_dim = self.get_hidden_size()

        # Project video embedding to transformer dim
        self.projection_layer = nn.Linear(input_dim, hidden_dim)

        # Load the embedding function for encoding token ids
        self.embedding_fn = load_embedding_fn(model_name, config_name)

        # Logit layer
        self.fc = nn.Sequential(nn.Linear(hidden_dim, fc_dim),
                                nn.Dropout(0.5),
                                nn.Tanh(),
                                nn.Linear(fc_dim, num_cls))

    def forward(self, video_emb, token_seq_ids, attention_mask):
        """
        # max_seq_len = max_video_len + num_special_tokens

        :param video_emb: [batch, max_video_len, video_emb_dim]
        :param token_seq_ids: [batch, max_seq_len]
        :param attention_mask: [batch, max_seq_len] <br>
        """
        # Project video embedding to token embedding space (hidden dim)
        video_emb = self.projection_layer(video_emb)

        # Encode video with positional embeddings
        video_emb = self.embedding_fn(inputs_embeds=video_emb,
                                      position_ids=torch.arange(1, self.max_len + 1, device=self.device))

        # Encode token sequence ([CLS] [UNK].. [SEP] [PAD]..)
        embeddings_input = self.embedding_fn(input_ids=token_seq_ids)

        # Replace [UNK] embeddings with video embeddings
        embeddings_input[:, 1: self.max_len+1, :] = video_emb

        # Extract the sequence embeddings from the final layer of the transformer
        last_hidden_states = self.transformer(inputs_embeds=embeddings_input,           # [batch, max_len, emb_dim]
                                              attention_mask=attention_mask)[0]

        # Obtain the CLS token embedding from the last hidden layer
        # [batch, emb_dim]
        cls_output = last_hidden_states[:, 0, :]

        logits = self.fc(cls_output)

        return logits

    def get_hidden_size(self):
        return self.transformer.config.hidden_size
