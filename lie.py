import torch
from scipy.stats import norm
from args import Arguments


def ataque_nao_oniciente(modelos:list, z = 1.5):
    args = Arguments()
    parametros = [[param.data.clone() for param in worker.parameters()] for worker in modelos[0:args.n_corrupted_workers]]

    grupos = list(zip(*parametros))
    # if len(modelos)//2 < wm:
    #     s = (args.n_workers // 2 + 1) - wm
    #     z = norm.ppf(1 - s / args.n_workers)
    # else:
    #     z = 1.5

    media = []
    desvios = []

    for grupo in grupos:
        empilhar = torch.stack(grupo, dim=0)
        media.append(torch.mean(empilhar, dim=0))
        desvios.append(torch.std(empilhar, dim=0))

    for worker in modelos[0:args.n_corrupted_workers]:
        for i, param in enumerate(worker.parameters()):
            valor = media[i] - z*desvios[i]
            param.data.copy_(valor)

def ataque_oniciente(modelos:list, z = 1.5):
    args = Arguments()
    parametros = [[param.data.clone() for param in worker.parameters()] for worker in modelos]

    grupos = list(zip(*parametros))
    # if len(modelos)//2 < wm:
    #     s = (args.n_workers // 2 + 1) - wm
    #     z = norm.ppf(1 - s / args.n_workers)
    # else:
    #     z = 1.5

    media = []
    desvios = []

    for grupo in grupos:
        empilhar = torch.stack(grupo, dim=0)
        media.append(torch.mean(empilhar, dim=0))
        desvios.append(torch.std(empilhar, dim=0))

    for worker in modelos[0:args.n_corrupted_workers]:
        for i, param in enumerate(worker.parameters()):
            valor = media[i] - z*desvios[i]
            param.data.copy_(valor)