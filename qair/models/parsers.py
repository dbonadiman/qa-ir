import torch
from qair.models.skeletons import Parser


@Parser.register('text')
class Text(Parser):

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def parse(self, ex):
        question_ids = list(map(lambda x: self.vocab[x], ex.question.lower().split()))
        passage_ids = list(map(lambda x: self.vocab[x], ex.passage.lower().split()))
        return (question_ids, passage_ids)

    def pad_to_batch(self, batch, max_len=None, pad_idx=0, leng=None):
        if max_len:
            max_len = max(5, min(max_len, max(len(x) for x in batch)))
        else:
            max_len = max(5, max(len(x) for x in batch))
        if leng:
            max_len = leng
        return [(max_len - len(x)) * [pad_idx] + x
                if len(x) < max_len
                else x[:max_len]
                for x in batch]

    def make_batch(self, docs, device='cpu'):
        question_ids, passage_ids = zip(*docs)
        q = torch.LongTensor(self.pad_to_batch(question_ids, pad_idx=self.vocab['PAD'])).to(device)
        a = torch.LongTensor(self.pad_to_batch(passage_ids, pad_idx=self.vocab['PAD'])).to(device)
        return q, a
        

@Parser.register('text_overlap')
class TextOverlap(Parser):

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab


    def parse(self, ex):
        question_ids = list(map(lambda x: self.vocab[x], ex.question.lower().split()))
        passage_ids = list(map(lambda x: self.vocab[x], ex.passage.lower().split()))
        o_question_ids = list(map(lambda x: 1 if x == '0' else 2, ex.question_overlap.split()))
        o_passage_ids = list(map(lambda x: 1 if x == '0' else 2, ex.passage_overlap.split()))
        assert len(question_ids) == len(o_question_ids)
        assert len(passage_ids) == len(o_passage_ids)
        return (question_ids, o_question_ids), (passage_ids, o_passage_ids)

    def pad_to_batch(self, batch, max_len=None, pad_idx=0):
        if max_len:
            max_len = max(5, min(max_len, max(len(x) for x in batch)))
        else:
            max_len = max(5, max(len(x) for x in batch))
        return [(max_len - len(x)) * [pad_idx] + x
                if len(x) < max_len
                else x[:max_len]
                for x in batch]

    def make_batch(self, docs, device='cpu'):
        question_ids, passage_ids = zip(*docs)
        q_i, qo_i = zip(*question_ids)
        p_i, po_i = zip(*passage_ids)
        q = torch.LongTensor(self.pad_to_batch(q_i, pad_idx=self.vocab['PAD'])).to(device)
        a = torch.LongTensor(self.pad_to_batch(p_i, pad_idx=self.vocab['PAD'])).to(device)
        q_o = torch.LongTensor(self.pad_to_batch(qo_i)).to(device)
        a_o = torch.LongTensor(self.pad_to_batch(po_i)).to(device)
        return (q, q_o), (a, a_o)