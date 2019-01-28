import argparse
import logging

import spacy
import re

from qair.data.example import Example
from qair.data.utils import create_path, list_files


class Tokenize:

    def __init__(self, ifile, nlp):
        logging.info(f'Processing File: {ifile.name}')
        self.reader = ifile
        self.nlp =  spacy.lang.en.English().Defaults.create_tokenizer(nlp)

    def normalize(self, sent):
        return re.sub(' +',' ',' '.join(sent)).split()


    def _spacy_process(self, text):
        return [w.text for w in self.nlp(text)]

    def read(self, row):
        return Example.from_json(row)

    def process(self, ex):
        q_tokens = self.normalize(self._spacy_process(ex.question))
        p_tokens = self.normalize(self._spacy_process(ex.passage))

        lower_q = list(map(lambda x: x.lower(), q_tokens))
        lower_p = list(map(lambda x: x.lower(), p_tokens))
        ex.feat('question', ' '.join(q_tokens))
        ex.feat('passage', ' '.join(p_tokens))
        ex.feat('question_overlap', ' '.join('1' if qt in lower_p else '0' for qt in lower_q))
        ex.feat('passage_overlap', ' '.join('1' if pt in lower_q else '0' for pt in lower_p))
        return ex

    def iterator(self, postprocess=lambda x: x):
        for row in self.reader:
            yield postprocess(self.process(self.read(row)))

    def to_file(self, filename):
        with open(create_path(filename), 'w') as out:
            out.writelines(self.iterator(lambda x: f"{x.to_json()}\n"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenize the dataset')
    parser.add_argument("dataset", help="dataset name")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    logging.info('-'*50)
    logging.info('Loading Spacy')
    nlp = spacy.load('en')
    logging.info('Spacy loaded')

    def process(dataset):
        input_path = f'data/clean/{dataset}/'
        output_path = f'data/parsed/{dataset}/'
        for inp_file, out_file in list_files(input_path, output_path):
                with open(inp_file) as ifile:
                    data = Tokenize(ifile, nlp)
                    data.to_file(out_file)


    process(args.dataset)