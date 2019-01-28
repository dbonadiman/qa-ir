from os.path import basename
from qair.data.example import Example
from qair.data.utils import create_path
import pickle

class QAdataset:

    def __init__(self, ifile, fold=0, train=True):
        self.reader = ifile
        self.dataset = ifile.split('/')[-2]
        self.qids = list(pickle.load(open(f'data/info/{self.dataset}/qids.pkl', 'rb'))[basename(ifile)])
        self.train = train
        if int(fold) > 0 and basename(ifile) == 'train.json' and self.train:
            self.fold = pickle.load(open(f'data/info/{self.dataset}/folds.pkl', 'rb'))[int(fold)]
            self.qids = set(self.qids) - set(self.fold)
        elif int(fold) > 0 and basename(ifile) == 'train.json':
            self.fold = pickle.load(open(f'data/info/{self.dataset}/folds.pkl', 'rb'))[int(fold)]
            self.qids = set(self.fold)
        self.qids = sorted(list(self.qids))  
        self.nb_questions = len(self.qids)
        self.q2e = {}
        self.list_ = []
        self.build(list(self.read_all()))

    


    def build(self, l):
        for qid in self.qids:
            self.q2e[qid] =  []
        for ex in l:
            if ex.qid in self.qids:
                self.q2e[ex.qid].append(ex)
            
    def read_all(self):
        with open(self.reader) as ifile:
            for row in ifile:
                yield self.read(row)

    def read(self, row):
        return Example.from_json(row)

    def rank(self, balanced=False):
        out =[]
        from itertools import product
        for qid in self.q2e:
            pos = []
            neg = []
            for ex in self.q2e[qid]:
                if ex.label == 1:
                    pos.append(ex)
                else:
                    neg.append(ex)
            if balanced:
                for pair in product(pos, neg[:5]):
                    out.append(pair)
            else:            
                for pair in product(pos, neg):
                    out.append(pair)
        return out

    def listwise(self, train=False):
        out = []
        for qid in self.q2e:
            if train:
                sm = sum(ex.label for ex in self.q2e[qid])
                if sm != 0:
                    out.append((qid, self.q2e[qid]))
            else:
                if len(self.q2e[qid]) > 0:
                    out.append((qid, self.q2e[qid]))
        return out



    @property
    def list(self):
        if self.list_:
            return self.list_
        for qid in self.q2e:
            for ex in self.q2e[qid]:
                self.list_.append(ex)
        return self.list_

    def iterator(self, postprocess=lambda x: x):
        for row in self.list:
            yield postprocess(row)

    def to_file(self, filename):
        with open(create_path(filename), 'w') as out:
            out.writelines(self.iterator(lambda x: f"{x.to_json()}\n"))