from scipy.stats import norm
from args import Arguments

args = Arguments()

for wm in range(0,51):
    s = (args.n_workers // 2 + 1) - wm
    z = norm.ppf(1 - s / args.n_workers)
    print(f'wm = {wm}   z = {z}')


