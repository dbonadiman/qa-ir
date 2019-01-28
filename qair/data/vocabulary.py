import argparse
from collections import Counter
from qair.data.utils import create_path, list_files
from qair.data.dataset import QAdataset
from itertools import chain
import logging

def giff_words(dataset, lower=True):

    def process(x):
        if lower:
            return x.lower().split()
        else:
            return x.split()

    for ex in dataset.iterator():
        for word in chain(process(ex.question), process(ex.passage)):
            yield word



if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    logging.info('-'*50)
    parser = argparse.ArgumentParser(description='Create the vocabulary')
    parser.add_argument("dataset", help="dataset name")
    parser.add_argument("--only_train", dest='onlytrain', help="lowercased", action='store_true')
    parser.add_argument('--lower', dest='lower', help="lowercased", action='store_true')
    parser.add_argument('--top_n', dest='n', help="the max number of words to keep", type=int)
    args = parser.parse_args()

    def process(dataset, lower=True):
        datasets = []
        inp_path = f'data/parsed/{dataset}/'
        for inp_file, _ in list_files(inp_path, inp_path):
            if not args.onlytrain or inp_file.endswith('train.json'):
                datasets.append(QAdataset(inp_file))
        return Counter(chain(*(giff_words(dataset, lower) for dataset in datasets)))

    vocabulary = process(args.dataset, args.lower)
    with open(create_path(f'data/info/{args.dataset}/vocab.tsv'), 'w') as ofile:
        top_n = vocabulary.most_common(args.n)
        logging.info(f'{args.dataset}: {len(top_n)} words')
        for word, freq in vocabulary.most_common(args.n):
            ofile.write(f'{word}\t{freq}\n')

