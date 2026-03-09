import numpy as np
import torch
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import lightgbm as lgb
from src.datasets.ardt_simple_dataset import ARDTSimpleDataset

class TabularARDTEstimator(BaseEstimator, RegressorMixin):
    """
    Combines classic tabular features with autoregressive context vectors to predict next token embeddings or labels.

    Parameters
    ----------
    word2idx : dict
        Mapping from token to index.
    embed_dim : int
        Embedding dimensionality.
    embeddings : np.ndarray, optional
        Pretrained embedding matrix (vocab_size x embed_dim).
    context_size : int, default=5
        Number of previous tokens to consider.
    alpha : float, default=0.5
        Exponential decay rate for context weighting.
    mode : {'classification', 'regression'}, default='regression'
        Whether to perform classification (next-token label) or regression (embedding prediction).
    model_type : {'decision_tree', 'random_forest', 'xgboost', 'lightgbm'}, default='lightgbm'
        Base estimator for MultiOutputRegressor.
    model_params : dict, optional
        Additional parameters to pass to the base estimator.
    tabular_params : dict, optional
        If provided, overrides model_params only for tabular features branch.
    """
    def __init__(
        self,
        word2idx,
        embed_dim=50,
        embeddings=None,
        context_size=5,
        alpha=0.5,
        mode='regression',
        model_type='lightgbm',
        model_params=None,
    ):
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}
        self.vocab_size = len(word2idx)
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.alpha = alpha
        self.mode = mode
        self.model_type = model_type
        self.model_params = model_params or {}

        if embeddings is not None:
            self.embeddings = embeddings
            self.embed_dim = embeddings.shape[1]
        else:
            self.embeddings = np.random.randn(self.vocab_size, self.embed_dim).astype(np.float32)

        base = self._init_base_estimator()
        self.model = MultiOutputRegressor(base)

    def _init_base_estimator(self):
        params = self.model_params.copy()
        if self.mode == 'classification':
            raise NotImplementedError("Classification mode not yet implemented for TabularARDTEstimator.")
        else:
            if self.model_type == 'decision_tree':
                from sklearn.tree import DecisionTreeRegressor
                return DecisionTreeRegressor(**params)
            elif self.model_type == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(**params)
            elif self.model_type == 'xgboost':
                return xgb.XGBRegressor(objective='reg:squarederror', **params)
            elif self.model_type == 'lightgbm':
                return lgb.LGBMRegressor(objective='regression', **params)
            else:
                raise ValueError(f"Unknown model_type={self.model_type}")

    def _compute_context_vector(self, tokens):
        """
        Weighted average of last context_size embeddings with exponential decay.
        """
        idxs = [self.word2idx.get(t) for t in tokens if t in self.word2idx]
        if not idxs:
            return np.zeros(self.embed_dim, dtype=np.float32)
        idxs = idxs[-self.context_size:]
        idxs_t = torch.tensor(idxs, dtype=torch.long)
        L = len(idxs_t)
        weights = torch.exp(-self.alpha * torch.arange(L-1, -1, -1, dtype=torch.float32))
        weights /= weights.sum()
        emb = torch.tensor(self.embeddings[idxs_t], dtype=torch.float32)
        ctx = torch.matmul(weights, emb)
        return ctx.numpy()

    def fit(self, X_tabular, sequences, y=None):
        """
        Fit the combined model.

        Parameters
        ----------
        X_tabular : array-like, shape (n_samples, n_tab_features)
            Numeric tabular features (e.g., genre encoded as one-hot or numeric).
        sequences : list of str or list of tokens
            Each element is a sequence (context) to build context vector from.
        y : array-like or None
            For regression: if provided, user-supplied target embeddings. Otherwise, we use next-token embeddings from sequences.
        """
        seq_dataset = ARDTSimpleDataset(sequences, self.word2idx, self.embeddings,
                                         context_size=self.context_size, alpha=self.alpha)
        seq_data = list(seq_dataset)
        X_seq = np.stack([x for x, t in seq_data])
        targets = np.array([t for x, t in seq_data])
        if self.mode == 'regression':
            Y = np.stack([self.embeddings[t] for t in targets])
        else:
            raise NotImplementedError("Classification not yet supported for TabularARDTEstimator.")

        X_tab = np.asarray(X_tabular)
        if X_tab.shape[0] != X_seq.shape[0]:
            raise ValueError("Number of tabular rows must match number of sequences.")

        X_comb = np.concatenate([X_tab, X_seq], axis=1)

        self.model.fit(X_comb, Y)
        return self

    def predict(self, X_tabular, sequences):
        """
        Predict next-token embeddings and return decoded token indices or embeddings.

        Returns
        -------
        predictions : list
            If mode='regression', returns next-token embeddings or decoded indices by argmax cosine.
        """
        X_tab = np.asarray(X_tabular)
        seq_vectors = np.stack([self._compute_context_vector(seq.split() if isinstance(seq, str) else seq)
                                 for seq in sequences])
        X_comb = np.concatenate([X_tab, seq_vectors], axis=1)

        preds = self.model.predict(X_comb)
        results = []
        for vec in preds:
            sims = cosine_similarity(vec.reshape(1, -1), self.embeddings)[0]
            idx = int(np.argmax(sims))
            results.append(self.idx2word[idx])
        return results

    def predict_proba(self, *args, **kwargs):
        raise NotImplementedError("Probability outputs not supported for regression mode.")

    def get_params(self, deep=True):
        return {
            'word2idx': self.word2idx,
            'embed_dim': self.embed_dim,
            'embeddings': self.embeddings,
            'context_size': self.context_size,
            'alpha': self.alpha,
            'mode': self.mode,
            'model_type': self.model_type,
            'model_params': self.model_params
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
