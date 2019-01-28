import json


class Example:

    def __init__(self, dict=None):
        if dict is None:
            dict = {}
        self._dict = dict

    def feat(self, key, feat):
        self._dict[key] = feat

    def process(self, key, func):
        self._dict[key] = func(self._dict[key])

    def __getattr__(self, name):
        return self._dict[name]

    def to_json(self):
        return json.dumps(self._dict)

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return str(self)

    @classmethod
    def from_json(cls, str):
        return cls(json.loads(str))
