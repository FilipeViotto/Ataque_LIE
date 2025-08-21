import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import norm
import matplotlib 
import matplotlib.pyplot as plt



# Classe Arguments para encapsular hiperparâmetros
class Arguments:
    def __init__(self):
        self.train_batch_size = 83
        self.test_batch_size = 83
        self.epochs = 13
        self.lr = 0.1                 # taxa de aprendizado, controla o tamanho da atualização dos parametros
        self.momentum = 0.9           # cria um termo de velocidade que influencia a atualização dos parametros
        self.l2 = 1e-4                # parametro de decaimento de peso, penalisa pesos muito grantes no treinamento
        self.n_workers = 51           # quantidade de dispositivos na simulação
        self.n_corrupted_workers = 50 # quantidade de dispositivos maliciosos
        self.no_cuda = False

args = Arguments()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self._f1 = torch.nn.Linear(28 * 28, 100)
        self._f2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self._f1(x.view(-1, 28 * 28)))
        x = torch.nn.functional.log_softmax(self._f2(x), dim=1)
        return x
    

def getdata():
    return MNIST(root="./data", train=True, download=True, transform=ToTensor()), MNIST(root="./data", train=False, download=True, transform=ToTensor())

def prepare_mnist_dataset():
    train_dataset, test_dataset = getdata()  # Obtém os datasets de treinamento e teste

    # Divisão do dataset de treinamento em partes para cada worker
    train_sets = []
    parte_size = len(train_dataset) // args.n_workers

    for i in range(args.n_workers):
        inicio = i * parte_size
        fim = (i + 1) * parte_size
        parte = Subset(train_dataset, range(inicio, fim))
        train_sets.append(parte)

    # Lidando com sobras de dados
    if len(train_dataset) % args.n_workers != 0:
        sobras = Subset(train_dataset, range(args.n_workers * parte_size, len(train_dataset)))
        train_sets.append(sobras)

    # Criação dos DataLoaders
    trainloaders = []
    for train_set in train_sets:
        trainloaders.append(DataLoader(train_set, batch_size=args.train_batch_size, shuffle = True))

    testloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

    return trainloaders, testloader

def train(args: Arguments, models: list, device, train_loaders, optimizers, criterion, epoch):

    for j in range(args.n_workers):
        models[j].train()
        for data, target in train_loaders[j]:
            data, target = data.to(device), target.to(device)
            output = models[j](data)
            loss = criterion(output, target)
            optimizers[j].zero_grad()
            loss.backward()

            # Clipping para evitar explosão de gradientes
            torch.nn.utils.clip_grad_norm_(models[j].parameters(), max_norm=1.0)
            optimizers[j].step()
        

def fedavg_aggregation(models: list, global_model):
    all_params = [list(model.parameters()) for model in models]
    average_params = [torch.stack(params).mean(0) for params in zip(*all_params)]

    for global_param, avg_param in zip(global_model.parameters(), average_params):
        global_param.data.copy_(avg_param)

    # Atualiza os modelos locais com os pesos do modelo global
    for model in models:
        model.load_state_dict(global_model.state_dict())


def test(global_model, device, test_loader, epoch, maior):

    global_model.eval()
    all_predictions = []
    all_targets = []
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = global_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_predictions.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())

    total_samples = len(test_loader.dataset)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted', zero_division=1)
    accuracy = (correct / total_samples) * 100
    if(accuracy>maior):
      maior = accuracy
      #print(f"Maior acurácia: {maior}")
    #with open(f"teste{wm}.txt", 'a') as file:
      #file.write(f'epoca {epoch} | Accuracy: {accuracy:.2f}% | Correct: {correct}/{total_samples} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1_score:.4f}\n')
    print(f'Teste: Precisão: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}, Acurácia: {accuracy:.2f}%')
    return maior


def ataque(workers_maliciosos:list, workers_total: int, z = 1.5):


    parametros = [[param.data.clone() for param in worker.parameters()]
                    for worker in workers_maliciosos]

    grupos = list(zip(*parametros))

    if len(workers_maliciosos)<workers_total//2 +1:
        s = (workers_total // 2 + 1) - len(workers_maliciosos)
        z = norm.ppf(1 - s / workers_total)
    else:
        z = norm.ppf(1 - 1 / workers_total)


    media = []
    desvios = []

    for grupo in grupos:
        empilhar = torch.stack(grupo, dim=0)
        media.append(torch.mean(empilhar, dim=0))
        desvios.append(torch.std(empilhar, dim=0))

    for worker in workers_maliciosos:
        for i, param in enumerate(worker.parameters()):
            valor = media[i] - z*desvios[i]
            param.data.copy_(valor)


def main(wm, maior):                                                                       # recebe a quantidade de workes maliciosos
    models = [MLP().to(device) for _ in range(args.n_workers)]                      # cria uma lista de workers
    optimizers = [optim.SGD(model.parameters(), lr= args.lr, momentum=args.momentum, weight_decay= args.l2) for model in models]  # SGD com momentum e L2 regularização (weight_decay)
    criterion = nn.CrossEntropyLoss()

    workers_maliciosos = []           # cria lista para armazenar os workes maliciosos
    #workers_maliciosos = random.sample(range(args.n_workers), args.n_corrupted_workers)
    for i in range(wm):               # marca os numeros dos workers maliciosos
        workers_maliciosos.append(i)

    # Criando os loaders de dados para cada trabalhador e o loader de teste global
    trainloaders, testloader = prepare_mnist_dataset()
    global_model = MLP().to(device)

    # Loop de treinamento federado
    for epoch in range(1, args.epochs + 1):
        train(args, models, device, trainloaders, optimizers, criterion, epoch)

        if epoch % 1 == 0:
            if wm>1:
                modelos_maliciosos = [models[i] for i in workers_maliciosos]
                ataque(modelos_maliciosos, args.n_workers)
            fedavg_aggregation(models, global_model)
            maior = test(global_model, device, testloader, epoch, maior)
    return maior

if __name__ == "__main__":
    lista_acuracia = []
    for wm in range(1,args.n_corrupted_workers+1):                  # passo com as quantidades de workers maliciosos em cada simulação
        print(f"Quantidade de workers maliciosos: {wm}")
        # with open(f"teste{wm}.txt", 'w') as file:
        #     file.write(f"Quantidade de workers maliciosos: {wm}\n")
        maior = main(wm, maior)
        # print(f"Maior acurácia: {maior}")
        lista_acuracia.append(maior)
        # with open(f"teste51.txt", 'a') as file:
        #     file.write(f"Maior acurácia: {maior}\n")
        #         
        plt.plot(range(1, len(lista_acuracia) + 1), lista_acuracia)
        plt.title("variacao na acuracia")
        plt.xlabel("Atacantes")
        plt.ylabel("Acuracia")
        plt.savefig('resultado')