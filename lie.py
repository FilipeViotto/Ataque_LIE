import torch
from scipy.stats import norm

def ataque(args,modelos:list,wm, z = 1.5):
    parametros = [[param.data.clone() for param in worker.parameters()] for worker in modelos[0:wm]]

    grupos = list(zip(*parametros))

    print(z)


    media = []
    desvios = []

    for grupo in grupos:
        empilhar = torch.stack(grupo, dim=0)
        media.append(torch.mean(empilhar, dim=0))
        desvios.append(torch.std(empilhar, dim=0))

    for worker in modelos[0:wm]:
        for i, param in enumerate(worker.parameters()):
            valor = media[i] - z*desvios[i]
            param.data.copy_(valor)