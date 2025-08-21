import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from sklearn.metrics import precision_recall_fscore_support
import random
import matplotlib.pyplot as plt
from scipy.stats import norm

from lie import ataque_nao_oniciente, ataque_oniciente
from modelo import MLP, MLP2
from args import Arguments
from processamento import prepare_dataset, fedavg_aggregation, train, test, prepare_mnist_dataset, main_train
import os


# Classe Arguments para encapsular hiperparâmetros

args = Arguments()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def main(wm):                                                                       # recebe a quantidade de workes maliciosos
    '''Função adicionada para teste. o comportamento da versão anterior estava inadequado'''
    lista_acuracia = []

    models = [MLP2().to(device) for _ in range(args.n_workers)]                      # cria uma lista de workers
    optimizers = [optim.SGD(model.parameters(), lr= args.lr, momentum=args.momentum, weight_decay= args.l2) for model in models]  # SGD com momentum e L2 regularização (weight_decay)
    criterion = nn.CrossEntropyLoss()

    workers_maliciosos = []           # cria lista para armazenar os workes maliciosos
    #workers_maliciosos = random.sample(range(args.n_workers), args.n_corrupted_workers)
    for i in range(wm):               # marca os numeros dos workers maliciosos
        workers_maliciosos.append(i)

    # Criando os loaders de dados para cada trabalhador e o loader de teste global
    trainloaders, testloader = prepare_mnist_dataset()
    global_model = MLP2().to(device)

    # Loop de treinamento federado
    for epoch in range(1, args.epochs + 1):
        print(epoch)
        main_train(args, models, device, trainloaders, optimizers, criterion, epoch)

        if epoch % 1 == 0:
            if wm>1:
                modelos_maliciosos = [models[i] for i in workers_maliciosos]
                ataque_nao_oniciente(models)
            fedavg_aggregation(models, global_model)
            maior, lista_acuracia = test(global_model, device, testloader, epoch, lista_acuracia)
    return maior, lista_acuracia


# Função principal
def gerenciamento(wm):
    maior = 0                                                   # define o maior como zero
    lista_acuracia = []
    models = [MLP().to(device) for _ in range(args.n_workers)]
    optimizers = [optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay) for model in models]
    loss_fns = [F.cross_entropy for _ in range(args.n_workers)]

    workers_maliciosos = []
    #workers_maliciosos = random.sample(range(args.n_workers), args.n_corrupted_workers)
    for i in range(0,wm):
        workers_maliciosos.append(i)

    # Criando os loaders de dados para cada trabalhador e o loader de teste global
    trainloaders, testloader = prepare_dataset()
    global_model = MLP().to(device)

    # Loop de treinamento federado
    for epoch in range(1, args.epochs + 1):
        print(f"epoca: {epoch}")
        train(args, models, device, trainloaders, optimizers, loss_fns, epoch)



        if epoch % 1 == 0:
            if wm>1:
                ataque_nao_oniciente(args, models, wm)
            fedavg_aggregation(models, global_model)
            maior, lista_acuracia = test(global_model, device, testloader, epoch, lista_acuracia)
    return maior, lista_acuracia


if __name__ == "__main__":
    lista_maior_acuracia = []
    x = []

    for wm in range(11,args.n_corrupted_workers+1):                  # passo com as quantidades de workers maliciosos em cada simulação
        x.append(wm)
        if not os.path.isdir(f'simulacoes'):
            os.makedirs(f'simulacoes')
        print(f"Quantidade de workers maliciosos: {wm}")
        # with open(f"teste{wm}.txt", 'w') as file:
        #   file.write(f"Quantidade de workers maliciosos: {wm}\n")
        maior, lista_de_acuracia = main(wm)



        # print(f"media_acuracia: {soma_acuracia/150}")
        # print(f"Maior acurácia: {maior}")
        lista_maior_acuracia.append(maior)

        plt.plot(range(args.epochs), lista_de_acuracia)
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.title(f'simulation{wm}')
        plt.savefig(f'simulacoes/simulation{wm}')
        plt.close()


        with open(f"teste51.txt", 'a') as file:
            file.write(f"Maior acurácia: {maior}\n")
        plt.plot(x,[round(numero, 3) for numero in lista_maior_acuracia])
        plt.xlabel('Quantidade de workers maliciosos')
        plt.ylabel('Acurácia')
        plt.title('Acurácia por quantidade de workers maliciosos')
        plt.savefig(f'intermediario.png')
        plt.close()

