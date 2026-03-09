import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator
from cuml.ensemble import RandomForestClassifier as cuRF


class ARDTCoTClassifier(BaseEstimator):
    def __init__(
        self,
        word2idx,
        embeddings,
        context_size=5,
        alpha=0.7,
        model_type='xgboost',
        model_params=None,
        depth=5,
        use_gpu=False
    ):
        self.word2idx = word2idx
        self.embeddings = embeddings
        self.context_size = context_size
        self.alpha = alpha
        self.embed_dim = embeddings.shape[1]
        self.model_type = model_type
        self.model_params = model_params or {}
        self.depth = depth
        self.use_gpu = use_gpu

    def _compute_context_vector(self, tokens):
        idxs = [self.word2idx.get(t) for t in tokens if t in self.word2idx]
        if not idxs:
            return np.zeros(self.embed_dim, dtype=np.float32)
        L = len(idxs)
        weights = np.exp(-self.alpha * np.arange(L-1, -1, -1, dtype=np.float32))
        weights /= weights.sum()
        emb = self.embeddings[idxs]
        return np.average(emb, axis=0, weights=weights)

    def _windowed_examples(self, text, label):
        tokens = text.split()
        examples = []
        for i in range(len(tokens) - self.context_size + 1):
            window = tokens[i:i + self.context_size]
            ctx_vec = self._compute_context_vector(window)
            examples.append((ctx_vec, label))
        return examples

    def fit(self, texts, labels):
        all_X, all_y = [], []
        for text, label in zip(texts, labels):
            ex = self._windowed_examples(text, label)
            for x, y in ex:
                all_X.append(x)
                all_y.append(y)

        X = np.stack(all_X).astype(np.float32)
        y = np.array(all_y)

        self._X_train = X
        self._y_train = y

        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                **self.model_params
            )

        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                device='gpu' if self.use_gpu else 'cpu',
                **self.model_params
            )

        elif self.model_type == 'random_forest':
            if self.use_gpu:
                self.model = cuRF(**self.model_params)
            else:
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(**self.model_params)

        elif self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(max_depth=self.depth, **self.model_params)

        elif self.model_type == 'adaboost':
            if self.use_gpu:
                self.model = xgb.XGBClassifier(
                    booster='gbtree',
                    tree_method='gpu_hist',
                    max_depth=1,
                    **self.model_params
                )
            else:
                base = DecisionTreeClassifier(max_depth=1)
                self.model = AdaBoostClassifier(base_estimator=base, **self.model_params)

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        self.model.fit(X, y)
        return self

    def predict(self, texts):
        preds = []
        for text in texts:
            windows = self._windowed_examples(text, label=0)
            X = np.stack([x for x, _ in windows]).astype(np.float32)
            probs = self.model.predict_proba(X)
            mean_prob = np.mean(probs, axis=0)
            preds.append(np.argmax(mean_prob))
        return np.array(preds)

    def predict_proba(self, texts):
        probas = []
        for text in texts:
            windows = self._windowed_examples(text, label=0)
            X = np.stack([x for x, _ in windows]).astype(np.float32)
            probs = self.model.predict_proba(X)
            mean_prob = np.mean(probs, axis=0)
            probas.append(mean_prob)
        return np.stack(probas)

    def explain_cot(self, text):
        if not hasattr(self.model, 'tree_'):
            raise ValueError("Model must be a DecisionTreeClassifier to use path tracing")

        windows = self._windowed_examples(text, label=0)
        X = np.stack([x for x, _ in windows]).astype(np.float32)

        paths = []
        tree = self.model.tree_
        for x in X:
            node = 0
            path = []
            while tree.feature[node] != -2 and len(path) < self.depth:
                feat = tree.feature[node]
                thresh = tree.threshold[node]
                if x[feat] <= thresh:
                    path.append(0)
                    node = tree.children_left[node]
                else:
                    path.append(1)
                    node = tree.children_right[node]
            path += [0] * (self.depth - len(path))
            paths.append(path)
        return paths
