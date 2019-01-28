import torch
import torch.nn as nn
from ..data import utils
from copy import deepcopy


class Model(nn.Module):

    _register = dict()

    def __init__(self):
        super().__init__()
        self.stored_checkpoint = None

    def trainable_parameters(self):
        return list(filter(lambda x: x.requires_grad, self.parameters()))

    def batched_iter(self, dataset, batch_size=32):
        ln = len(dataset)
        for idx in range(0, ln, batch_size):
            yield dataset[idx:min(ln, idx+batch_size)]

    def checkpoint(self):
        self.stored_checkpoint = deepcopy(self.state_dict())

    def load_checkpoint(self):
        self.load_state_dict(self.stored_checkpoint)

    def save(self, path):
        path = utils.create_path(path)
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    @classmethod
    def register(cls, name):

        def add_to_register(subclass):
            if name in Model._register:
                raise Exception(f'Duplicate subcalss {name}')
            Model._register[name] = subclass

        return add_to_register
    
    @classmethod
    def by_name(cls, name):
        if name not in Model._register:
            raise Exception(f'Invalid model name: {name}')
        return Model._register.get(name)

    @classmethod
    def list_available(cls):
        keys = list(cls._register.keys())
        return [k for k in keys]


class Trainer:

    _register = dict()

    def __init__(self, parser, model):
        self.parser = parser
        self.model = model

    @classmethod
    def register(cls, name):

        def add_to_register(subclass):
            if name in Trainer._register:
                raise Exception(f'Duplicate subcalss {name}')
            Trainer._register[name] = subclass

        return add_to_register
    
    @classmethod
    def by_name(cls, name):
        if name not in Trainer._register:
            raise Exception(f'Invalid model name: {name}')
        return Trainer._register.get(name)

    @classmethod
    def list_available(cls):
        keys = list(cls._register.keys())
        return [k for k in keys]



class Parser:

    _register = dict()

    @classmethod
    def register(cls, name):

        def add_to_register(subclass):
            if name in Parser._register:
                raise Exception(f'Duplicate subcalss {name}')
            Parser._register[name] = subclass

        return add_to_register
    
    @classmethod
    def by_name(cls, name):
        if name not in Parser._register:
            raise Exception(f'Invalid model name: {name}')
        return Parser._register.get(name)

    @classmethod
    def list_available(cls):
        keys = list(cls._register.keys())
        return [k for k in keys]

