from typing import List
import argparse
from collections import OrderedDict

import torch
from torch import nn, arange, cat, optim
import torch.nn.functional as F
import horovod.torch as hvd

class EnumType:
    def __init__(self, name, maxCategories, embed_dim=0):
        self.name = name
        self.maxCategories = maxCategories
        self.embed_dim = embed_dim


class EmbeddingsCollection(nn.Module):
    def __init__(self, annotation, concat=True, use_batch_norm=True):
        super().__init__()

        self.output_size = 0
        self.embeddings = {}
        for k, embed in annotation.items():
            self.embeddings[k] = torch.nn.Embedding(
                embed.maxCategories,
                embed.embed_dim,
                sparse=True,
            )
            self.output_size += embed.embed_dim
        self.embeddings = torch.nn.ModuleDict(self.embeddings)
        self.concat = concat

        if use_batch_norm and concat:
            self.input_normalizer = torch.nn.BatchNorm1d(self.output_size)
        else:
            self.input_normalizer = None

    def forward(self, input: OrderedDict):
        embeds = []
        k: str
        v: nn.modules.sparse.Embedding
        for k, v in self.embeddings.items():
            embeds.append(v(input[k]))
        if self.concat:
            embeds = torch.cat(embeds, -1)
        if self.input_normalizer:
            return self.input_normalizer(embeds)
        return embeds


class OneHotEncoding(nn.Module):
    def __init__(self, maxSize):
        super().__init__()
        assert isinstance(maxSize, int)
        self.maxSize = maxSize

    def forward(self, x):
        result = torch.zeros(*x.shape, self.maxSize, device=x.device, dtype=torch.float32)
        result.scatter_(-1, x.unsqueeze(-1), 1.0)
        return result


class OneHotEncodingCollection(nn.Module):
    def __init__(self, annotation, use_batch_norm=True):
        super().__init__()

        self.output_size = 0
        self.one_hot = {}
        for k, embed in annotation.items():
            self.one_hot[k] = OneHotEncoding(embed.maxCategories)
            self.output_size += embed.maxCategories

        self.one_hot = torch.nn.ModuleDict(self.one_hot)

    def forward(self, input: OrderedDict):
        embeds = []
        k: str
        v: OneHotEncoding
        for k, v in self.one_hot.items():
            embeds.append(v(input[k]))
        return embeds


class MyInput(nn.Module):
    def __init__(self, annotation, use_bn=False):
        super().__init__()

        self.output_size = 0
        if 'embeddings' in annotation:
            self.embeddings = EmbeddingsCollection(annotation["embeddings"], use_batch_norm=use_bn)
            self.output_size += self.embeddings.output_size

        if 'one_hot' in annotation:
            self.one_hot = OneHotEncodingCollection(annotation["one_hot"], use_batch_norm=use_bn)
            self.output_size += self.one_hot.output_size

        if self.output_size == 0:
            raise ValueError("MyInput was not able to process " + str(annotation.keys()))

    def forward(self, buffer: OrderedDict):
        features: List[torch.Tensor] = []

        if hasattr(self, "embeddings"):
            features.append(self.embeddings(buffer["embeddings"]))

        if hasattr(self, "one_hot"):
            features += self.one_hot(buffer['one_hot'])

        return torch.cat(features, -1)


class MyLinear(nn.Linear):
    def forward(self, x):
        if len(self.weight.shape) == 2:
            return super().forward(x)
        else:
            return torch.baddbmm(self.bias.unsqueeze(-2), x, self.weight.transpose(-1, -2))


class MySequential(nn.Module):
    def __init__(self, input_size, layers=None,
                 use_dropout=False, dropout_rate=0.5, use_bn=False,
                 activation=nn.ReLU, activate_final=False):
        super().__init__()
        self.layers = []

        if isinstance(input_size, (tuple, list)):
            input_size, = input_size

        if not isinstance(activation, (tuple, list)):
            activation = [activation] * len(layers)
            if not activate_final:
                activation[-1] = None

        hidden_dim = input_size
        for l_dim, act in zip(layers or [], activation):
            self.layers.append(MyLinear(hidden_dim, l_dim))
            if act is not None:
                if use_bn:
                    self.layers.append(nn.BatchNorm1d(l_dim))
                self.layers.append(act())
                if use_dropout:
                    self.layers.append(nn.Dropout(dropout_rate))
            hidden_dim = l_dim
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inputs):
        return self.layers(inputs)


class MyModel(nn.Module):
    def __init__(self, annotation: OrderedDict, layers=(512, 1), layer_size=512, num_heads=4,
                 t_size=128, use_dropout=False, use_bn=True, num_segments=48):

        super().__init__()
        annotation = annotation
        if not layers:
            layers = [layer_size, layer_size, 1]
        self.num_segments = num_segments
        self.input = MyInput(annotation)

        self.sequential = MySequential(self.input.output_size, layers=layers, use_dropout=use_dropout, use_bn=use_bn)
        self.sequential2 = MySequential(layers[-1]+self.num_segments+2, layers=[1], use_dropout=use_dropout, use_bn=use_bn,)

    def forward(self, buffer):
        hot1 = buffer["one_hot"]["hot1"]
        hot1_ref = arange(self.num_segments).reshape(1, self.num_segments).float()
        hot0 = buffer["one_hot"]["hot0"]
        hot0_ref = arange(2).reshape(1, 2).float()
        if hot1.is_cuda:
            hot1_ref = hot1_ref.cuda()
            hot0_ref = hot0_ref.cuda()
        hot1_onehot = (hot1.float().unsqueeze(-1) == hot1_ref).float()
        hot0_onehot = (hot0.float().unsqueeze(-1) == hot0_ref).float()
        network = self.sequential(self.input(buffer))
        link = self.sequential2(cat([network, hot1_onehot, hot0_onehot], dim=2))
        return link.squeeze(-1)


def huber_loss(a, b, delta=20):
    err = (a - b).abs()
    mask = err < delta
    return (0.5 * mask * (err ** 2)) + ~mask * (err * delta - 0.5 * (delta ** 2))


annotation = OrderedDict()
annotation["embeddings"] = OrderedDict()
annotation["embeddings"]["name0"] = EnumType("name0", 2385, 12)
annotation["embeddings"]["name1"] = EnumType("name1", 201, 8)
annotation["embeddings"]["name2"] = EnumType("name2", 201, 8)
annotation["embeddings"]["name3"] = EnumType("name3", 6, 3)
annotation["embeddings"]["name4"] = EnumType("name4", 19, 5)
annotation["embeddings"]["name5"] = EnumType("name5", 1441, 11)
annotation["embeddings"]["name6"] = EnumType("name6", 201, 8)
annotation["embeddings"]["name7"] = EnumType("name7", 22, 5)
annotation["embeddings"]["name8"] = EnumType("name8", 156, 8)
annotation["embeddings"]["name9"] = EnumType("name9", 1216, 11)
annotation["embeddings"]["name10"] = EnumType("name10", 9216, 14)
annotation["embeddings"]["name11"] = EnumType("name11", 88999, 17)
annotation["embeddings"]["name12"] = EnumType("name12", 941792, 20)
annotation["embeddings"]["name13"] = EnumType("name13", 9405, 14)
annotation["embeddings"]["name14"] = EnumType("name14", 83332, 17)
annotation["embeddings"]["name15"] = EnumType("name15", 828767, 20)
annotation["embeddings"]["name16"] = EnumType("name16", 945195, 20)

annotation["one_hot"] = OrderedDict()
annotation["one_hot"]["hot0"] = EnumType("hot0", 3) # one_hot doesn't use dimension
annotation["one_hot"]["hot1"] = EnumType("hot1", 50) # one_hot doesn't use dimension
