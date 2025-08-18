import args
args = args.Arguments()
from modelo import MLP

def TrimmedMean(modelos: list):
    modelo_resultante = MLP()
    lista_parametros_aberta = []
    for modelo in modelos:
        lista_parametros_aberta = modelo
