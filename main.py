import torch
import torch.nn as nn
from modules.add import MyAddModule

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.add = MyAddModule()

    def forward(self, input1, input2):
        return self.add(input1, input2)

model = MyNetwork()
input1, input2 = torch.randn(5, 5), torch.randn(5, 5)
print(model(input1, input2))
print(input1 + input2)
