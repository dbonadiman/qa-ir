import pickle

dict_ = pickle.load(open('data/raw/yahooqa/env.pkl', 'rb'))
for dataset in ['train', 'dev', 'test']:
    pickle.dump(dict_[dataset], open(f'data/raw/yahooqa/{dataset}.pkl', 'wb'))