import torch
import torch.nn as nn

from qair.models import Model
from qair.models.layers import KimConv, activations

torch.backends.cudnn.deterministic = True

@Model.register('cnn')
class CNN(Model):

    def __init__(self, params, vocab, device='cpu'):
        super().__init__()
        self.vocab = vocab

        self.device = torch.device(device)
        params['emb_num'], params['emb_dim'] = vocab.shape

        self.embs = nn.Embedding(params['emb_num'],
                                 params['emb_dim'],
                                 vocab['PAD'])

        self.conv_q = KimConv(params['emb_dim'],
                              params['qcnn']['conv_size'],
                              windows=params['qcnn']['windows'],
                              activation=activations[params['qcnn']['activation']])

        self.conv_a = KimConv(params['emb_dim'],
                            params['acnn']['conv_size'],
                            windows=params['acnn']['windows'],
                            activation=activations[params['acnn']['activation']])

        self.mlp = nn.Sequential(
            nn.Linear(params['qcnn']['conv_size']+params['acnn']['conv_size'], params['hidden_size']),
            nn.Tanh(),
            nn.Linear(params['hidden_size'], 1),
        )

        self.embs.weight.data.copy_(torch.from_numpy(vocab.weights))
        if 'static_emb' in params and params['static_emb']:
            self.embs.weight.requires_grad = False

    def forward(self, inp):
        (q, a) = inp
        # Map ids to word embeddings
        q = self.embs(q)
        a = self.embs(a)

        # Convolutional Encoder

        qemb = self.conv_q(q)
        aemb = self.conv_a(a)

        # sim = self.bilin(qemb, aemb)
        # Concat the two branches
        join = torch.cat([qemb, aemb], -1)

        # A multi layer perceptron computes the output score
        out = self.mlp(join)

        return out

@Model.register('severyn_2016')
class Severyn16(Model):

    def __init__(self, params, vocab, device='cpu'):
        super().__init__()
        self.vocab = vocab

        self.device = torch.device(device)
        params['emb_num'], params['emb_dim'] = vocab.shape
        self.embs = nn.Embedding(params['emb_num'],
                                 params['emb_dim'],
                                 vocab['PAD'])
        self.ov_embs = nn.Embedding(3, 5, 0)

        self.conv_q = KimConv(params['emb_dim']+5,
                              params['qcnn']['conv_size'],
                              windows=params['qcnn']['windows'],
                              activation=activations[params['qcnn']['activation']])

        self.conv_a = KimConv(params['emb_dim']+5,
                            params['acnn']['conv_size'],
                            windows=params['acnn']['windows'],
                            activation=activations[params['acnn']['activation']])

        self.bilin = nn.Bilinear(params['qcnn']['conv_size'],
                                 params['acnn']['conv_size'],
                                 1,
                                 bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(params['qcnn']['conv_size']+params['acnn']['conv_size']+1, params['hidden_size']),
            nn.Tanh(),
            nn.Linear(params['hidden_size'], 1)
        )

        self.embs.weight.data.copy_(torch.from_numpy(vocab.weights))
        if 'static_emb' in params and params['static_emb']:
            self.embs.weight.requires_grad = False

    def forward(self, inp):
        ((q, q_o), (a, a_o)) = inp
        # Map ids to word embeddings
        q = self.embs(q)
        a = self.embs(a)

        # Map overlap ids to overlap embeddings
        q_o = self.ov_embs(q_o)
        a_o = self.ov_embs(a_o)

        # Create a unique token

        q = torch.cat([q, q_o], -1)
        a = torch.cat([a, a_o], -1)

        # Convolutional Encoder
        qemb = self.conv_q(q)
        aemb = self.conv_a(a)

        sim = self.bilin(qemb, aemb)
        # Concat the two branches
        join = torch.cat([qemb, sim, aemb], -1)

        # A multi layer perceptron computes the output score
        out = self.mlp(join)
        return out

@Model.register('relcnn')
class RelCNN(Model):

    def __init__(self, params, vocab, device='cpu'):
        super().__init__()
        self.vocab = vocab

        self.device = torch.device(device)
        params['emb_num'], params['emb_dim'] = vocab.shape
        self.embs = nn.Embedding(params['emb_num'],
                                 params['emb_dim'],
                                 vocab['PAD'])
        self.ov_embs = nn.Embedding(3, 5, 0)

        self.conv_q = KimConv(params['emb_dim']+5,
                              params['qcnn']['conv_size'],
                              windows=params['qcnn']['windows'],
                              activation=activations[params['qcnn']['activation']])

        self.conv_a = KimConv(params['emb_dim']+5,
                            params['acnn']['conv_size'],
                            windows=params['acnn']['windows'],
                            activation=activations[params['acnn']['activation']])

        self.mlp = nn.Sequential(
            nn.Linear(params['qcnn']['conv_size']+params['acnn']['conv_size'], params['hidden_size']),
            nn.Tanh(),
            nn.Linear(params['hidden_size'], 1)
        )

        self.embs.weight.data.copy_(torch.from_numpy(vocab.weights))
        if 'static_emb' in params and params['static_emb']:
            self.embs.weight.requires_grad = False

    def forward(self, inp):
        ((q, q_o), (a, a_o)) = inp
        # Map ids to word embeddings
        q = self.embs(q)
        a = self.embs(a)

        # Map overlap ids to overlap embeddings
        q_o = self.ov_embs(q_o)
        a_o = self.ov_embs(a_o)

        # Create a unique token

        q = torch.cat([q, q_o], -1)
        a = torch.cat([a, a_o], -1)

        # Convolutional Encoder
        qemb = self.conv_q(q)
        aemb = self.conv_a(a)

        # sim = self.bilin(qemb, aemb)
        # Concat the two branches
        join = torch.cat([qemb, aemb], -1)

        # A multi layer perceptron computes the output score
        out = self.mlp(join)
        return out
