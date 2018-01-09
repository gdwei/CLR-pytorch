import torch
import torch.utils.data.sampler as sampler

class SubsetSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        # super(SubsetSampler, self).__init__()
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples
