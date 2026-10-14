# JAM3 gene expression data

The data are 36 CSV files, one per tissue (`0001.csv`, `0005.csv`, ...). Each file holds one dataset in the transfer learning setup:

- The first (unnamed) column is the sample index.
- The `JAM3` column is the response: expression of the JAM3 gene.
- The remaining 399 columns are the predictors: expression of the other genes. They are the same in every file.

These are the data used for the real-data example in the preprint. They generally follow the GTEx JAM2 application of Li, Cai & Li (2022), "Transfer learning for high-dimensional linear regression", but for different tissues/genes.

## Download

The data are distributed as a release asset rather than kept in git, so `pip install` stays small:

```
wget https://github.com/NathanWycoff/bayes_trans/releases/download/v0.2.0/proc_data.tar.gz
echo "a2f3ac3a1a89db1af2dd958b56d4a1c4ec3332329dca9228a696b55a717916c5  proc_data.tar.gz" | sha256sum -c
tar xzf proc_data.tar.gz   # -> proc_data/*.csv (about 148 MB)
```

## Loading

```python
from glob import glob
import numpy as np
import pandas as pd

dfs = [pd.read_csv(f, index_col=0) for f in sorted(glob('proc_data/*.csv'))]
Xs = [np.array(df.iloc[:, 1:]) for df in dfs]
ys = [np.array(df.iloc[:, 0]) for df in dfs]
```

Remember to center and scale `X` and `y` before calling `bayes_trans`.
