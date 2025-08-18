import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden_layer = nn.Linear(784, 100)  # 784 features de entrada, 100 neurônios ocultos
        self.output_layer = nn.Linear(100, 10)   # 100 neurônios ocultos, 10 classes de saída

    def forward(self, x):
        x = x.view(-1, 784)  # Achatar a entrada
        x = F.relu(self.hidden_layer(x))  # Aplicar ReLU na camada oculta
        x = self.output_layer(x)          # Camada de saída (sem softmax aqui, aplicado na função de perda)
        x = F.softmax(x, dim=1)           # usa softmax aqui para obter a probabilidade das saidas
        return x