import torch
import torch.nn.functional as F
from torch.cuda import is_available
from torch.nn import Embedding, Linear, LSTM, LSTMCell, Module
device = torch.device("cuda" if is_available() else "cpu")


class Encoder(Module):
    def __init__(self, inp_dim, 
                 embed_dim,
                 enc_units):
        '''
        h_i - hidden vectors
        state_c - cell state output
        state_h - hidden state output
        '''
        super().__init__()
        self.embedding = Embedding(num_embeddings=inp_dim, 
                                   embedding_dim=embed_dim,
                                   padding_idx=0)
        
        self.lstm = LSTM(embed_dim, 
                         enc_units, 
                         num_layers=2,
                         batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        x, state = self.lstm(x)
        state_h, state_c = state
        return (x, state_h[-1], state_c[-1])


class DeepComEncoder(Module):
    def __init__(self, inp_dim_code, 
                 inp_dim_ast, 
                 embed_dim=512, 
                 enc_units=128):
        '''
        contains both code 
        and AST encoder
        '''
        super().__init__()
        self.encoder_code = Encoder(inp_dim_code, embed_dim, enc_units)
        self.encoder_ast = Encoder(inp_dim_ast, embed_dim, enc_units)
        self.W1 = Linear(2*enc_units, enc_units)
        self.W2 = Linear(2*enc_units, enc_units)
        self.W3 = Linear(2*enc_units, enc_units)

    def forward(self, x_c, x_a):
        '''
        x_c = batch x timestep
        x_a = batch x timestep
        '''
        hidden_c, state_c_h, state_c_c = self.encoder_code(x_c)
        hidden_a, state_a_h, state_a_c = self.encoder_code(x_a)
        hidden = self.W1(torch.cat((hidden_c, hidden_a), dim=-1))
        state_h = self.W2(torch.cat((state_c_h, state_a_h), dim=-1))
        state_c = self.W3(torch.cat((state_c_c, state_a_c), dim=-1))
        return (hidden, (state_h, state_c))


class BahdanauAttention(Module):
    def __init__(self, attn_sz):
        '''
        attn_sz = hidden[2], the decoder units
        '''
        super().__init__()

        self.V = Linear(attn_sz, 1, bias=False)
        self.Ws = Linear(2*attn_sz, attn_sz) 
        self.Wh = Linear(attn_sz, attn_sz, bias=False) 
        self.Wc = Linear(1, attn_sz, bias=False)                 

    def forward(self, h_i, s_t, coverage=None):
        '''
        h_i - encoder states - batch_sz x timesteps x enc_units
        s_t - decoder state - batch_sz x dec_units
        '''
        enc_features = self.Wh(h_i)                         
        dec_features = self.Ws(s_t)
        dec_features = dec_features.unsqueeze(1)
        features = enc_features + dec_features 
    
        if coverage is not None:
            cov_features = coverage.unsqueeze(-1)
            cov_features = self.Wc(cov_features)
            features = features + cov_features
            
        e_t = self.V(torch.tanh(features))
        e_t = e_t.squeeze(-1)
        a_t = F.softmax(e_t, dim=-1)
        
        if coverage is None:
            coverage = a_t
        else:
            coverage = coverage + a_t

        context_vector = torch.bmm(a_t.unsqueeze(1), h_i)
        context_vector = context_vector.squeeze(1)
        return (context_vector, a_t, coverage)


class AttentionDecoder(Module):
    def __init__(self, inp_dim,
                 embed_dim=512, 
                 dec_units=128):
        '''
        attn_shape is same as enc_out_shape: h_i shape
        inp_dim == out_dim == 30001
        '''
        super().__init__()
        self.attention = BahdanauAttention(dec_units)
        self.embedding = Embedding(num_embeddings=inp_dim, 
                                   embedding_dim=embed_dim, 
                                   padding_idx=0)
        
        self.lstm = LSTMCell(embed_dim, dec_units)
        self.W1 = Linear((embed_dim + dec_units), embed_dim)
        self.W2 = Linear((3*dec_units + embed_dim), 1)
        self.V1 = Linear(2*dec_units, 3*dec_units)
        self.V2 = Linear(3*dec_units, inp_dim)

    def forward(self, x, h_i, prev_state, context_vector, coverage, max_oovs, code_ext):
        if context_vector is None:
            s_t = torch.cat(prev_state, 1)
            context_vector, _, _ = self.attention(h_i, s_t)
        
        x = self.embedding(x).squeeze(1)             
        x1 = self.W1(torch.cat((x, context_vector), 1))
        
        cell_out, state = self.lstm(x1, prev_state)
        s_t = torch.cat((cell_out, state), 1)
        
        context_vector, attn_dist, coverage = self.attention(h_i, s_t, coverage)
        p_vocab = torch.cat((cell_out, context_vector), 1)
        p_vocab = self.V1(p_vocab)
        p_vocab = self.V2(p_vocab)
        p_vocab = F.softmax(p_vocab, dim=1)

        #-----------------------------------**pointer-gen**--------------------------------#
        
        batch_sz = list(x.size())[0]
        p_gen = torch.sigmoid(self.W2(torch.cat((context_vector, s_t, x), 1)))
        
        p_vocab = p_gen * p_vocab
        attn_dist = (1.0 - p_gen) * attn_dist
        
        extra_zeros = torch.zeros(batch_sz, max_oovs, device=device)
        p_vocab = torch.cat((p_vocab, extra_zeros), axis=-1)
        final_dist = p_vocab.scatter_add(dim=-1, index=code_ext, src=attn_dist)
        return (final_dist, attn_dist, (cell_out, state), coverage, context_vector)