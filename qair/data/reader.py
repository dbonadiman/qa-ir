import logging
import re
import argparse
import sys
import pickle

from os.path import basename
from qair.data.example import Example
from qair.data.utils import create_path, list_files


class wikiqa:

    def __init__(self, ifile):
        logging.info(f'Processing WikiQA: {ifile.name}')
        self.reader = list(ifile)[1:]
        logging.info(len(self.reader))
        self.qids = set()

    def read(self, row):
        ex = Example()
        ex.feat('qid', row[0])
        ex.feat('question', str(row[1]))
        ex.feat('passage', str(row[5]))
        ex.feat('label', int(row[6]))
        return ex

    def iterator(self, postprocess=lambda x: x):
        for row in self.reader:
            ex = self.read(row.strip().split('\t'))
            self.qids.add(ex.qid)
            yield postprocess(ex)

    def to_file(self, filename):
        with open(create_path(filename), 'w') as out:
            out.writelines(self.iterator(lambda x: f"{x.to_json()}\n"))


class yahooqa:

    def __init__(self, ifile):
        logging.info(f'Processing YahooQA: {ifile.name}')
        self.reader = pickle.load(ifile)
        logging.info(len(self.reader))
        self.qids = set()

    def read(self, row):
        ex = Example()
        ex.feat('qid', row[0])
        ex.feat('question', str(row[1]))
        ex.feat('passage', str(row[2]))
        ex.feat('label', int(row[3]))
        return ex

    def iterator(self, postprocess=lambda x: x):
        for qid, question in enumerate(self.reader):
            self.qids.add(qid)
            for answer, label in self.reader[question]:
                ex = self.read((qid, question, answer, label))
                yield postprocess(ex)

    def to_file(self, filename):
        with open(create_path(filename), 'w') as out:
            out.writelines(self.iterator(lambda x: f"{x.to_json()}\n"))



class trecqa:

    def __init__(self, ifile):
        logging.info(f'Processing TrecQA: {ifile.name}')
        self.reader = ifile
        self.qids = set()

    def read(self, row):
        ex = Example()
        ex.feat('qid', str(row[0]))
        ex.feat('question', str(row[1]))
        ex.feat('passage', str(row[2]))
        ex.feat('label', int(row[3]))
        return ex

    def iterator(self, postprocess=lambda x: x):
        prev = ''
        for line in self.reader:
            line = line.strip()
            qid_match = re.match('<QApairs id=\'(.*)\'>', line)
            if qid_match:
                qid = qid_match.group(1)
                self.qids.add(qid)
            if prev and prev.startswith('<question>'):
                question = ' '.join(line.split('\t'))
            label = re.match('^<(positive|negative)>', prev)
            if label:
                label = label.group(1)
                label = 1 if label == 'positive' else 0
                answer = ' '.join(line.split('\t'))
                yield postprocess(self.read((qid, question, answer, label)))
            prev = line

    def to_file(self, filename):
        with open(create_path(filename), 'w') as out:
            out.writelines(self.iterator(lambda x: f"{x.to_json()}\n"))



if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='Clean the dataset')
    parser.add_argument("dataset", help="dataset name")
    parser.add_argument("train", help="train")
    parser.add_argument("dev", help="dev")
    parser.add_argument("test", help="test")
    parser.add_argument("--byte",type=bool, default=False)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    logging.info('-'*50)

    def process(dataset, file_names=None):
        files_counter = {}
        input_path = f'data/raw/{dataset}/'
        output_path = f'data/clean/{dataset}/'
        for inp_file, out_file in list_files(input_path, output_path, file_names):
            with open(inp_file) if not args.byte else open(inp_file, 'rb') as ifile:
                data = getattr(sys.modules[__name__], dataset)(ifile)
                data.to_file(out_file)
                files_counter[basename(out_file)] = data.qids
        # logging.info(files_counter)
        pickle.dump(files_counter, open(create_path(f'data/info/{dataset}/qids.pkl'), 'wb'))
                
    
    process(args.dataset, {args.train:'train.json', args.dev:'dev.json', args.test:'test.json'})
