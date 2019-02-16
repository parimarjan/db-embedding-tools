import numpy as np
import json
import utils
import sys
import torch
from torch.autograd import Variable
import random

class ReplayMemory():

    def __init__(self, N, load_existing=False, data_dir="./data"):
        self.max = N
        self.memory = []

    def add(self, experience):
        '''
        This operation adds a new experience e, replacing the earliest experience if arrays are full.
        '''
        if len(self.memory) == self.max:
            # then will need to replace something
            self.memory = self.memory[1:-1]
        assert len(self.memory) < self.max, 'len must be less'
        self.memory.append(experience)

    def sample(self, size):
        '''
        Samples slice of arrays with input size.
        '''
        return random.sample(self.memory, size)

    def can_sample(self, sample_size):
        '''
        Returns true if item count is at least as big as sample size.
        '''
        return len(self.memory) >= sample_size
