import torch
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from sklearn.metrics import precision_recall_fscore_support
from args import Arguments
args = Arguments()

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
        trainloaders.append(DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True))

    testloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    return trainloaders, testloader

# Função para preparar datasets divididos para os trabalhadores (sem mudanças necessárias)
def prepare_dataset():
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = MNIST(root='./data', train=False, transform=transform)

    trainloaders = []
    subset_size = len(train_dataset) // args.n_workers
    for i in range(args.n_workers):
        worker_subset = Subset(train_dataset, range(i * subset_size, (i + 1) * subset_size))
        trainloader = DataLoader(worker_subset, batch_size=args.train_batch_size, shuffle=True)
        trainloaders.append(trainloader)

    testloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False) # Lote maior para teste

    return trainloaders, testloader

# Função de treinamento local para cada trabalhador
def train(args: Arguments, models: list, device, train_loaders, optimizers, loss_fns, epoch):
    loss = [0] * args.n_workers

    for j in range(args.n_workers):
        models[j].train()
        for data, target in train_loaders[j]:
            data, target = data.to(device), target.to(device)
            optimizers[j].zero_grad()
            output = models[j](data)
            loss[j] = loss_fns[j](output, target)

            loss[j].backward()

            # Clipping para evitar explosão de gradientes
            torch.nn.utils.clip_grad_norm_(models[j].parameters(), max_norm=1.0)
            optimizers[j].step()

# Função de agregação FedAvg
def fedavg_aggregation(models: list, global_model):
    all_params = [list(model.parameters()) for model in models]
    average_params = [torch.stack(params).mean(0) for params in zip(*all_params)]

    for global_param, avg_param in zip(global_model.parameters(), average_params):
        global_param.data.copy_(avg_param)

    # Atualiza os modelos locais com os pesos do modelo global
    for model in models:
        model.load_state_dict(global_model.state_dict())

# Função de teste para o modelo global
def test(global_model, device, test_loader, epoch, lista_acuracia):
    maior = 0
    soma_acuracia = 0
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
    lista_acuracia.append(accuracy)
    soma_acuracia = soma_acuracia + accuracy
    if(accuracy>maior):
      maior = accuracy
    #   print(f"Maior acurácia: {maior}")
    # with open(f"teste{wm}.txt", 'a') as file:
    #   file.write(f'epoca {epoch} | Accuracy: {accuracy:.2f}% | Correct: {correct}/{total_samples} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1_score:.4f}\n')
    # #print(f'Teste: Precisão: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}, Acurácia: {accuracy:.2f}%')

    return maior, lista_acuracia


def main_train(args: Arguments, models: list, device, train_loaders, optimizers, criterion, epoch):

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