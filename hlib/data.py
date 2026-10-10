"""Dataset splitting and containers for features, targets and model splits."""

import numpy as np


def split_dataset_train_val_test(df, seed=42, pct=0.2):
    n = len(df)
    n_val = int(n * pct)
    n_test = int(n * pct)
    n_train = n - n_val - n_test

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    df_test = df.iloc[idx[n_train + n_val:]].reset_index(drop=True)

    return df_train, df_val, df_test


def split_feature_target_columns(
    df, target_columns: list[str], training_columns: list[str] | None = None
):
    if training_columns is None:
        training_columns = [c for c in df.columns if c not in target_columns]

    S = IOData(df[training_columns], df[target_columns])
    return S


class IOData:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        self.to_matrices()

    def __repr__(self):
        return f"Features: {self.features}\nTargets: {self.targets}"

    def to_matrices(self):
        self.X = self.features.to_numpy()
        self.y = self.targets.to_numpy()
        self.Xe = np.column_stack([np.ones(self.X.shape[0]), self.X])


class ModelData:
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test

    @classmethod
    def from_dataframe(
        cls,
        df,
        target_columns: list[str],
        training_columns: list[str] | None = None,
        seed: int = 42,
        pct: float = 0.2,
    ):
        if not 0 < pct < 0.5:
            raise ValueError("pct deve essere tra 0 e 0.5 esclusi")

        splits = split_dataset_train_val_test(df, seed=seed, pct=pct)

        train, val, test = (
            split_feature_target_columns(
                subset,
                target_columns=target_columns,
                training_columns=training_columns,
            )
            for subset in splits
        )

        return cls(train=train, val=val, test=test)
