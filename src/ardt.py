import numpy as np
import torch
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import lightgbm as lgb
import nltk
from src.datasets.ardt_simple_dataset import ARDTSimpleDataset
from utils.handler_functions import compute_target_embedding
nltk.download('punkt', quiet=True)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class ARDT:
    def __init__(
        self,
        word2idx,
        embed_dim=50,
        mode='classification',       
        context_size=None,
        alpha=0.5,
        model_type='decision_tree',  
        nested_params=None,  
        embeddings=None
    ):
        self.word2idx   = word2idx
        self.idx2word   = {i:w for w,i in word2idx.items()}
        self.vocab_size = len(word2idx)
        self.mode       = mode
        self.context_size = context_size
        self.alpha      = alpha

        if embeddings is not None:
            self.embeddings = embeddings
            self.embed_dim  = embeddings.shape[1]
        else:
            self.embed_dim  = embed_dim
            self.embeddings = np.random.randn(self.vocab_size, self.embed_dim).astype(np.float32)


        if mode == 'classification':
            if model_type == 'decision_tree':
                if nested_params is not None:
                    self.model = DecisionTreeClassifier(**nested_params)
                else:
                    self.model = DecisionTreeClassifier()
            elif model_type == 'random_forest':
                if nested_params is not None:
                    self.model = RandomForestClassifier(**nested_params)
                else:
                    self.model = RandomForestClassifier(n_estimators=50)
            elif model_type == 'xgboost':
                self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
                if nested_params is not None:
                    self.model = xgb.XGBClassifier(**nested_params)
                else:
                    self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            elif model_type == 'lightgbm':
                self.model = lgb.LGBMClassifier(objective='multiclass', num_class=self.vocab_size)
                if nested_params is not None:
                    self.model = lgb.LGBMClassifier(**nested_params)
                else:
                    self.model = lgb.LGBMClassifier(objective='multiclass', num_class=self.vocab_size)
            else:
                raise ValueError(f"Unknown model_type={model_type}")

        else:  
            if model_type == 'decision_tree':
                if nested_params is not None:
                    base = DecisionTreeRegressor(**nested_params)
                else:
                    base = DecisionTreeRegressor()
            elif model_type == 'random_forest':
                if nested_params is not None:
                    base = RandomForestRegressor(**nested_params)
                else:
                    base = RandomForestRegressor(n_estimators=50)
            elif model_type == 'xgboost':
                if nested_params is not None:
                    base = xgb.XGBRegressor(**nested_params)
                else:
                    base = xgb.XGBRegressor(objective='reg:squarederror')
            elif model_type == 'lightgbm':
                base = lgb.LGBMRegressor(objective='regression')
                if nested_params is not None:
                    base = lgb.LGBMRegressor(**nested_params)
                else:
                    base = lgb.LGBMRegressor(objective='regression')
            else:
                raise ValueError(f"Unknown model_type={model_type}")
            self.model = MultiOutputRegressor(base)

    def _compute_context_vector(self, tokens):
        idxs = [self.word2idx.get(t) for t in tokens if t in self.word2idx]
        if not idxs:
            return np.zeros(self.embed_dim, dtype=np.float32)
        idxs = idxs[-self.context_size:] if self.context_size else idxs
        idxs = torch.tensor(idxs, dtype=torch.long)
        L = len(idxs)
        weights = torch.exp(-self.alpha * torch.arange(L-1, -1, -1, dtype=torch.float32))
        weights /= weights.sum()
        emb = torch.tensor(self.embeddings[idxs], dtype=torch.float32)
        ctx_vec = torch.matmul(weights, emb)
        return ctx_vec.numpy()

    def fit(self, dataset):
        """
        dataset yields (context_vec, target_idx).
        In classification: target_idx is integer label.
        In regression: we use embedding[target_idx] as vector y.
        """                
        X, y = zip(*dataset)  
        X = np.stack(X)
        y = np.stack([self.embeddings[t] if self.mode == 'regression' else t for t in y])

        print(f"[INFO] Fitting ARDT model: X shape {X.shape}, y shape {y.shape}")
        self.model.fit(X, y)

    def predict_next(self, context, return_proba=False):
        tokens = context.split() if isinstance(context, str) else list(context)
        ctx_vec = self._compute_context_vector(tokens).reshape(1, -1)

        if self.mode == 'classification':
            probs = self.model.predict_proba(ctx_vec)[0]
            idx   = int(np.argmax(probs))
            word  = self.idx2word[idx]
            return (word, probs) if return_proba else (word, None)
        else:
            pred_vec = self.model.predict(ctx_vec)[0]           
            sims = cosine_similarity(pred_vec.reshape(1,-1), self.embeddings[:self.vocab_size])[0]
            idx_in_model = int(np.argmax(sims))
            word = self.idx2word.get(idx_in_model, "<UNK>")
            print(f"Length of idx: {len(self.idx2word)}, ")
            return (word, sims) if return_proba else (word, None)
        

    def generate(self, prompt, max_len=20):
        tokens = prompt.split() if isinstance(prompt, str) else list(prompt)
        for _ in range(max_len):
            w, _ = self.predict_next(tokens, return_proba=False)
            tokens.append(w)
        return " ".join(tokens)

    def compute_perplexity(self, dataset):
        if self.mode!='classification':
            raise NotImplementedError("Perplexity only for classification.")
        logp = []
        for x_vec, label in dataset:
            probs = self.model.predict_proba([x_vec])[0]
            p     = max(probs[label], 1e-12)
            logp.append(np.log(p))
        return float(np.exp(-np.mean(logp)))

    def compute_bleu(self, references, hypotheses):
        refs = [[r.split()] for r in references]
        hyps = [h.split() for h in hypotheses]
        smoothie = SmoothingFunction().method1
        return np.mean([sentence_bleu(r, h, smoothing_function=smoothie) for r, h in zip(refs, hyps)])


    def get_path_embedding(self, context_vec, max_depth=None):
        """
        Returns a binary vector representing the path through the decision tree
        for a given input context vector.
        """
        if not hasattr(self.model, "tree_"):
            raise ValueError("Model is not a decision tree or not yet trained.")

        tree = self.model.tree_
        node = 0  
        path = []

        depth = tree.max_depth if max_depth is None else max_depth
        while tree.feature[node] != -2 and (len(path) < depth):
            feature_index = tree.feature[node]
            threshold = tree.threshold[node]
            if context_vec[feature_index] <= threshold:
                path.append(0)
                node = tree.children_left[node]
            else:
                path.append(1)
                node = tree.children_right[node]

        if len(path) < depth:
            path += [0] * (depth - len(path))

        return np.array(path[:depth], dtype=np.float32)
    
    

class ARDTEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, word2idx=None, embed_dim=50, embeddings=None,
                 model_type='lightgbm', context_size=5, alpha=0.5, model_params=None):
        self.word2idx = word2idx
        self.embed_dim = embed_dim
        self.embeddings = embeddings
        self.model_type = model_type
        self.context_size = context_size
        self.alpha = alpha
        self.model_params = model_params if model_params is not None else {}

    def fit(self, texts, y=None):
        dataset = ARDTSimpleDataset(texts, self.word2idx, self.embeddings,
                                    context_size=self.context_size, alpha=self.alpha)
        
        data = [(x, y) for x, y in dataset]

        self.ardt_ = ARDT(
            word2idx=self.word2idx,
            embeddings=self.embeddings,
            model_type=self.model_type,
            context_size=self.context_size,
            alpha=self.alpha,
            mode='regression',
            nested_params=self.model_params
        )
        self.ardt_.fit(data) 
        return self

    def predict(self, texts):
        if isinstance(texts, np.ndarray) or isinstance(texts[0], (np.ndarray, list)):
            context_vectors = np.vstack(texts)
        else:
            context_vectors = np.stack([
                self.ardt_._compute_context_vector(t.split()) for t in texts
            ])

        pred_vecs = self.ardt_.model.predict(context_vectors)
        
        return np.array([
            compute_target_embedding(
                self.ardt_.idx2word[np.argmax(cosine_similarity(p.reshape(1,-1), self.ardt_.embeddings[:self.ardt_.vocab_size])[0])],
                self.ardt_.embeddings,
                self.ardt_.word2idx
            ) for p in pred_vecs
        ])
        
    def predict_proba(self, texts):
        """
        texts : list of context strings
        Returns: list of probability arrays over the entire vocab
        """
        probas = []
        for context in texts:
            _, probs = self.ardt_.predict_next(context, return_proba=True)
            probas.append(probs)
        return np.vstack(probas)

    def get_params(self, deep=True):
        return {
            'word2idx': self.word2idx,
            'embed_dim': self.embed_dim,
            'embeddings': self.embeddings,
            'model_type': self.model_type,
            'context_size': self.context_size,
            'alpha': self.alpha,
            'model_params': self.model_params,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def score(self, X, y):
        print(f"[INFO] Scoring with ARDTEstimator: X shape {len(X)}, y shape {len(y)}")
        y_pred = self.predict(X)
        print(f"[INFO] y_pred shape: {y_pred.shape}")
        
        if y_pred.shape != y.shape:
            print(f"[WARNING] Shape mismatch in score(): y_pred {y_pred.shape}, y {y.shape}")
            return float("nan")


        sims = cosine_similarity(y_pred, y)
        score = np.mean(np.diag(sims))

        if np.isnan(score):
            print("[WARNING] Cosine similarity returned NaN")
        return score