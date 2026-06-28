import torchvision.transforms.v2 as T
import random

class RandomGausiannBlur(T.GaussianBlur):
    def __init__(self, kernel_size, sigma):
        super().__init__(kernel_size, sigma)
        self.rand_threshold = 0.5
        self.distrib = random.uniform(0,1)
    
    def forward(self, *inputs):
        if self.distrib < self.rand_threshold:
            return super().forward(*inputs)
        else: 
            return inputs[0] if len(inputs) == 1 else inputs

