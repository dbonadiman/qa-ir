import logging
import argparse
import pickle
import random


def slice_list(lst, n):

    sz = len(lst)//n
    print(len(lst), sz, sz*n)
    return [lst[i:i+sz] for i in range(0, len(lst), sz)]

if __name__=='__main__':
    random.seed(123)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    logging.info('-'*50)
    parser = argparse.ArgumentParser(description='Create the vocabulary')
    parser.add_argument("dataset", help="dataset name")
    parser.add_argument("--n", help="dataset name", default=10)
    args = parser.parse_args()
    folds = {}
    with open(f'data/info/{args.dataset}/qids.pkl', 'rb') as iflie:
        qids = pickle.load(iflie)['train.json']
        qids = list(qids)
        random.shuffle(qids)
        
        for i, sub in enumerate(slice_list(qids, 10)[:10], 1):
            folds[i] = sub

    with open(f'data/info/{args.dataset}/folds.pkl', 'wb') as ofile:
        pickle.dump(folds, ofile)
        
    