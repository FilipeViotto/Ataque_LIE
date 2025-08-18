class Arguments:
    def __init__(self):
        self.train_batch_size = 83  # Conforme o artigo
        self.test_batch_size = 83 # Tamanho do lote de teste maior para avaliação mais eficiente
        self.epochs = 20
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.n_workers = 51
        self.n_corrupted_workers = 12 # 24% dos trabalhadores são corrompidos (12/51)
        self.no_cuda = False