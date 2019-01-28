import argparse
import numpy as np
import logging
from gensim.models import KeyedVectors
from qair.data.utils import create_path
np.random.seed(666)

def filter_embeddings(vocab, emb_dict):
    for word, _ in vocab:
        if word in emb_dict:
            yield (word, emb_dict[word])
        else:
            logging.debug(word)

def random_from_vocab(vocab, mean, std, dim):
    for word, _ in vocab:
        yield (word, np.random.normal(mean, std, (dim,)))


def emb_stats(embs_dict, vocab):
    e = np.array([embs_dict[w] for w in embs_dict])
    return float(np.mean(e)), float(np.std(e)), e.shape[1]

def load_vocabulary(ifile):
    lst = []
    with open(ifile) as inp:
        for line in inp:
            word, cnt = line.strip().split()
            lst.append((word, cnt))
    return lst

def load_w2v_fast(ifile, vocab_set):
    dct = {}
    with open(ifile) as inp:
        next(inp)
        for line in inp:
            line = line.strip()
            word, emb = line.split(' ', 1) 
            if word in vocab_set:
                dct[word] = np.fromstring(emb, sep=' ')
    return dct

class Embeddings:

    def __init__(self, ifile):
        embs = KeyedVectors.load_word2vec_format(ifile, binary=(ifile[-3:] == 'bin'))
        self.w2idx = d={x:i for i,x in enumerate(embs.vocab)}
        self.emb_weights = np.array([embs[w] for w in embs.vocab])
    
    def __getitem__(self, word):
        return self.w2idx[word] if word in self.w2idx else self.w2idx['UNK']
    
    @property
    def shape(self):
        return self.emb_weights.shape

    @property
    def weights(self):
        return self.emb_weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create embedding')
    parser.add_argument("dataset", help="dataset name")
    parser.add_argument('embeddings', help="embedding name")
    parser.add_argument('--dim', dest='dim', help='embedding dimension ignored if embeddings is not rand', type=int)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info('-'*50)

    vocab = load_vocabulary(f'data/info/{args.dataset}/vocab.tsv')
    if args.embeddings == 'rand.txt':
        if args.dim is None:
            raise Exception('dim argument required')
        mean, std, dim = 0., 0.5, args.dim
        filtered_w2v = dict(random_from_vocab(vocab, mean, std, dim))
        filtered_w2v['PAD'] = np.zeros((dim,))
        filtered_w2v['UNK'] = np.random.normal(mean, std, (dim,))
        
    elif args.embeddings[-3:] == 'bin':
        w2v = KeyedVectors.load_word2vec_format(f'embs/{args.embeddings}', binary=(args.embeddings[-3:] == 'bin'))
        mean, std, dim = emb_stats(w2v, w2v.vocab)
        logging.info(f'Stats Mean {mean} Std {std} Dim {dim}')
        logging.info(f'Original Vocab: {len(vocab)}')
        logging.info(f'Embedding Vocab: {len(w2v.vocab)}')
        filtered_w2v = dict(filter_embeddings(vocab, w2v))
        logging.info(f'Filtered Vocab: {len(filtered_w2v)}')


        filtered_w2v['PAD'] = np.zeros((dim,))
        filtered_w2v['UNK'] = np.random.normal(mean, std, (dim,))
    else:
        vocab_set = set(word for word, _ in vocab)
        w2v = load_w2v_fast(f'embs/{args.embeddings}', vocab_set)
        
        mean, std, dim = emb_stats(w2v, w2v)
        logging.info(f'Stats Mean {mean} Std {std} Dim {dim}')
        logging.info(f'Original Vocab: {len(vocab)}')
        logging.info(f'Embedding Vocab: {len(w2v)}')
        filtered_w2v = dict(filter_embeddings(vocab, w2v))
        logging.info(f'Filtered Vocab: {len(filtered_w2v)}')


        filtered_w2v['PAD'] = np.zeros((dim,))
        filtered_w2v['UNK'] = np.random.normal(mean, std, (dim,))

    logging.info('Saving file...')
    with open(create_path(f'data/embs/{args.dataset}/{args.embeddings[:-4]}.txt'), 'w') as ofile:
        ofile.write(f'{len(filtered_w2v)} {dim}\n')
        for word, emb  in filtered_w2v.items():
            ofile.write(f"{word} {' '.join(str(val) for val in emb)}\n")
