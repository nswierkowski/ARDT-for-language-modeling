import gc
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBRegressor
from src.ardt import ARDT, ARDTEstimator
from src.datasets.ardt_simple_dataset import ARDTSimpleDataset
import numpy as np
from sklearn.model_selection import train_test_split
from src.datasets.audio_dataset import AudioARDTDataset
from utils.handler_functions import build_vocab, compute_target_embedding, load_glove_embeddings, load_gpt2_embeddings, load_minilm_embeddings
from datasets import load_dataset, DownloadConfig
import yaml
import joblib
import pickle
import time
import json

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
import scipy.stats as stats
from sklearn.multioutput import MultiOutputRegressor
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import librosa
from scipy.io.wavfile import write as wav_write
from sklearn.model_selection import KFold
from itertools import product


def save_yaml(obj, path):
    with open(path, 'w') as f:
        yaml.dump(obj, f)


def load_texts(cfg):
    if 'dataset_name' in cfg:
        ds = load_dataset(cfg['dataset_name'], split=cfg.get('dataset_split', 'train'))
        column = cfg.get('dataset_column', 'text')
        texts = ds[cfg.get('dataset_field', column)]
    else:
        with open(cfg['texts'], encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    if 'sample_size' in cfg:
        texts = texts[:cfg['sample_size']]
    if 'num_records' in cfg:
        texts = texts[:cfg['num_records']]
    return texts

HYPERPARAM_DISTS = {
    'decision_tree': {
        'model_params__max_depth': stats.randint(3, 20),
        'model_params__min_samples_split': stats.randint(2, 20),
    },
    'random_forest': {
        'model_params__n_estimators': stats.randint(10, 200),
        'model_params__max_depth': stats.randint(3, 20),
    },
    'xgboost': {
        'model_params__n_estimators': stats.randint(50, 300),
        'model_params__max_depth': stats.randint(3, 10),
        'model_params__learning_rate': stats.loguniform(1e-3, 1e-1),
    },
    'lightgbm': {
        'model_params__n_estimators': stats.randint(50, 150),       
        'model_params__learning_rate': stats.loguniform(1e-3, 1e-1), 
        'model_params__num_leaves': stats.randint(20, 50),           
    }
}


def nested_train_model(config_path):
    cfg = yaml.safe_load(open(config_path))['train_model']
    model_type = cfg.get('model_type', 'lightgbm')
    model_dir = Path(cfg['model_dir']); model_dir.mkdir(parents=True, exist_ok=True)

    texts = load_texts(cfg)
    vocab = build_vocab(texts)

    emb_type = cfg.get("embedding_type", "glove")
    if emb_type == "gpt2":
        embeddings = load_gpt2_embeddings(vocab)
    elif emb_type == "minilm":
        embeddings = load_minilm_embeddings(vocab)
    else:
        embeddings = load_glove_embeddings(cfg['embedding'], vocab, embed_dim=cfg['model_params'].get('embed_dim', 50))

    embed_dim = embeddings.shape[1]

    param_dist = {
        'context_size': stats.randint(5, 100),
        'alpha': stats.uniform(0.1, 1.0),
        
       # For lightgbm
       # 'model_params__n_estimators': stats.randint(50, 200),
       # 'model_params__max_depth': stats.randint(3, 10),
       # 'model_params__learning_rate': stats.loguniform(1e-3, 1e-1),
       
       # For Decision Tree
        # 'model_params_max_depth': stats.randint(3, 20),
        # 'model_params_min_samples_split': stats.randint(2, 20),
        # "model_params_min_samples_leaf": stats.randint(2, 20),
        # "cpp_alpha": stats.uniform(0.1, 1.0),
        
        # For Random Forest
        'model_params__n_estimators': stats.randint(10, 200),
        'model_params__max_depth': stats.randint(3, 20),
        'model_params__min_samples_split': stats.randint(2, 20),
        'model_params__min_samples_leaf': stats.randint(1, 20),
        
        # For XGBoost
        # 'model_params__n_estimators': stats.randint(50, 300),
        # 'model_params__max_depth': stats.randint(3, 10),
        # 'model_params__learning_rate': stats.loguniform(1e-3, 1e-1),
    }
    
    param_dist.update(HYPERPARAM_DISTS.get(model_type, {}))

    true_y = np.array([compute_target_embedding(text, embeddings, vocab) for text in texts])
    base_estimator = ARDTEstimator(
        word2idx=vocab,
        embed_dim=embed_dim,
        embeddings=embeddings,
        model_type=model_type,
    )

    search = HalvingRandomSearchCV(
        estimator=base_estimator,
        param_distributions=param_dist,
        factor=2,
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    search.fit(texts, true_y)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    result = {
        'best_params': best_params,
        'best_score': best_score
    }

    with open(model_dir / 'nested_cv_results.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"[DONE] Best model score: {best_score:.4f}")
    print(f"[DONE] Results saved to {model_dir / 'nested_cv_results.json'}")


def train_model(config_path):
    cfg = yaml.safe_load(open(config_path))['train_model']
    model_type = cfg.get('model_type', 'lightgbm')
    model_dir = Path(cfg['model_dir']); model_dir.mkdir(parents=True, exist_ok=True)
    data_dir  = Path(cfg['dataset_dir']); data_dir.mkdir(parents=True, exist_ok=True)

    texts = load_texts(cfg)
    train_texts, test_texts = train_test_split(texts, test_size=cfg.get('test_size', 0.2), random_state=42)

    vocab = build_vocab(train_texts)
    
    
    ardt = ARDT(word2idx=vocab, model_type=model_type, **cfg.get('model_params', {}))
    
    emb_type = cfg.get("embedding_type", "glove")
    if emb_type == "gpt2":
        ardt.embeddings = load_gpt2_embeddings(vocab)
    elif emb_type == "minilm":
        ardt.embeddings = load_minilm_embeddings(vocab, cache_path="cached_minilm_embeddings.npy")
    elif emb_type == "tree":
        from src.tree_embedding.embedding import iterative_ardt
        iterative_ardt(
            X=np.array([vocab[word] for word in vocab.keys()]).reshape(-1, 1),
            y=np.array(list(vocab.values())),
            n_iters=cfg.get('tree_embedding_iters', 3),
            max_depth=cfg.get('tree_embedding_max_depth', None)
        )
    else:
        ardt.embeddings = load_glove_embeddings(cfg['embedding'], vocab, embed_dim=ardt.embed_dim)
        

    def build_xy(list_texts):
        ds = ARDTSimpleDataset(list_texts, vocab, ardt.embeddings,
                                context_size=ardt.context_size, alpha=ardt.alpha)
        X = np.vstack([x for x, _ in ds])
        if ardt.mode == 'classification':
            y = np.array([y for _, y in ds])
        else:
            y = np.vstack([ardt.embeddings[y] for _, y in ds])
        return X, y, ds

    X_train, y_train, train_ds = build_xy(train_texts)
    X_test,  y_true,  test_ds  = build_xy(test_texts)

    # tuning
    if cfg.get('tune', False):
        space = HYPERPARAM_DISTS.get(model_type, {})
        if isinstance(ardt.model, MultiOutputRegressor):
            space = {f'estimator__{k}': v for k, v in space.items()}
        search = HalvingRandomSearchCV(ardt.model, space, factor=2,
                                       random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        ardt.model = search.best_estimator_
        save_yaml(search.best_params_, model_dir / 'best_params.yaml')
    else:
        ardt.model.fit(X_train, y_train)

    y_pred = ardt.model.predict(X_test)
    y_true_norm = y_true / np.linalg.norm(y_true, axis=1, keepdims=True)
    y_pred_norm = y_pred / np.linalg.norm(y_pred, axis=1, keepdims=True)
    cos_sims = np.sum(y_true_norm * y_pred_norm, axis=1)
    mean_cos_sim = np.mean(cos_sims)
    std_cos_sim = np.std(cos_sims)

    joblib.dump(ardt.model, model_dir / 'ardt_model.pkl')
    np.save(model_dir / 'embeddings.npy', ardt.embeddings)
    save_yaml(vocab, model_dir / 'word2idx.yaml')
    save_yaml({
        'model_type': model_type,
        'mode': ardt.mode,
        'context_size': ardt.context_size,
        'alpha': ardt.alpha,
        'embed_dim': ardt.embed_dim
    }, model_dir / 'ardt_meta.yaml')
    pickle.dump(train_ds, open(data_dir / 'ardt_dataset.pkl', 'wb'))

    json.dump({
        'mean_cosine_similarity': mean_cos_sim,
        'std_cosine_similarity': std_cos_sim,
        'train_size': len(train_ds),
        'test_size': len(test_ds)
    }, open(model_dir / 'results.json', 'w'), indent=2)

    print(f"[INFO] Model saved to {model_dir}")
    print(f"[RESULT] Cosine Similarity: mean = {mean_cos_sim:.4f}, std = {std_cos_sim:.4f}")


def run_model(config_path):
    cfg = yaml.safe_load(open(config_path))['run_model']
    model_dir = Path(cfg['model_dir'])
    max_len   = cfg.get('max_length', 10)

    embeddings = np.load(model_dir / 'embeddings.npy')
    vocab      = yaml.safe_load(open(model_dir / 'word2idx.yaml'))
    meta       = yaml.safe_load(open(model_dir / 'ardt_meta.yaml'))
    model      = joblib.load(model_dir / 'ardt_model.pkl')

    ardt = ARDT(
        word2idx=vocab,
        model_type=meta['model_type'],
        mode=meta['mode'],
        context_size=meta['context_size'],
        alpha=meta['alpha'],
        embed_dim=meta['embed_dim'],
        embeddings=embeddings
    )
    ardt.model = model

    print("Interactive ARDT ready. ~max_len=N to adjust, ~exit to quit.")
    while True:
        inp = input('>> ').strip()
        if not inp:
            continue
        if inp.startswith('~'):
            if inp == '~exit':
                break
            if inp.startswith('~max_len='):
                max_len = int(inp.split('=')[1]); print(f"max_len={max_len}")
            continue
        start = time.time()
        out = ardt.generate(inp, max_len=max_len)
        dt = time.time() - start
        print(f"Prompt:    {inp}\nGenerated: {out}\nTime:      {dt:.3f}s\n")
        
def compute_mcd(mfcc1, mfcc2):
    """Compute Mel Cepstral Distortion (MCD) between two MFCC matrices"""
    if len(mfcc1) != len(mfcc2):
        min_len = min(len(mfcc1), len(mfcc2))
        mfcc1, mfcc2 = mfcc1[:min_len], mfcc2[:min_len]
    diff = mfcc1 - mfcc2
    sq_diff = np.square(diff)
    sum_sq_diff = np.sum(sq_diff, axis=1)
    mcd = (10.0 / np.log(10)) * np.sqrt(2) * np.mean(np.sqrt(sum_sq_diff))
    return mcd

def train_music_model_with_ardt(config_path):
    cfg = yaml.safe_load(open(config_path))['train_model']
    model_type = cfg.get('model_type', 'lightgbm')
    model_dir = Path(cfg['model_dir']); model_dir.mkdir(parents=True, exist_ok=True)
    data_dir  = Path(cfg['dataset_dir']); data_dir.mkdir(parents=True, exist_ok=True)

    genre_filter = cfg.get('genre', 'Instrumental')
    dataset_name = cfg.get('dataset_name', 'lewtun/music_genres_small')
    split = cfg.get('split', 'train')
    max_samples = cfg.get('max_samples', None)

    print("[INFO] Loading dataset...")
    dataset = load_dataset(dataset_name, split=split, cache_dir=str(data_dir), streaming=True)
    dataset = dataset.filter(lambda x: x['genre'] == genre_filter)
    if max_samples:
        dataset = dataset.select(range(max_samples))

    sr = cfg.get('sample_rate', 8000)
    max_len = cfg.get('max_length_seconds', 1)
    n_mfcc = cfg.get('n_mfcc', 13)
    context_size = cfg.get('context_size', 5)
    alpha = cfg.get('alpha', 0.5)

    X_list, y_list = [], []

    print("[INFO] Processing audio samples...")
    for i, sample in enumerate(dataset):
        audio = sample['audio']
        y = librosa.resample(np.array(audio['array']), orig_sr=audio['sampling_rate'], target_sr=sr)
        y = y[:sr * max_len]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T

        for j in range(context_size, len(mfcc)):
            context = mfcc[j-context_size:j]
            weights = np.exp(-alpha * np.arange(context_size-1, -1, -1))
            weights /= weights.sum()
            X_list.append(np.dot(weights, context))
            y_list.append(mfcc[j])

        del mfcc, y
        gc.collect()

        print(f"  Processed {i + 1} samples...")

    X = np.stack(X_list)
    y = np.stack(y_list)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.get('test_size', 0.1), random_state=42)

    print("[INFO] Initializing ARDT model...")
    dummy_vocab = {str(i): i for i in range(len(y_train))}
    dummy_embeddings = np.vstack([y_train, y_test]) 
    ardt = ARDT(
        word2idx=dummy_vocab,
        embed_dim=n_mfcc,
        mode='regression',
        context_size=context_size,
        alpha=alpha,
        model_type=model_type,
        embeddings=dummy_embeddings
    )

    print("[INFO] Fitting ARDT model...")
    dummy_dataset = list(zip(X_train, range(len(X_train))))  
    ardt.fit(dummy_dataset)

    print("[INFO] Evaluating ARDT model on test set...")
    y_pred = []
    for x in X_test:
        pred_vec, _ = ardt.predict_next(x, return_proba=False)
        pred_vec = ardt.embeddings[ardt.word2idx[pred_vec]] if isinstance(pred_vec, str) else pred_vec
        y_pred.append(pred_vec)
    y_pred = np.stack(y_pred)

    mcd_score = compute_mcd(y_test, y_pred)
    print(f"[RESULT] Mean Cepstral Distortion (MCD): {mcd_score:.4f}")

    results_path = model_dir / 'ardt_eval_results.json'
    results = {
        "dataset_name": dataset_name,
        "split": split,
        "genre_filter": genre_filter,
        "test_size": len(y_test),
        "sample_rate": sr,
        "n_mfcc": n_mfcc,
        "context_size": context_size,
        "alpha": alpha,
        "model_type": model_type,
        "mcd": mcd_score,
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Evaluation results saved to {results_path}")


def run_music_model(config_path):
    cfg = yaml.safe_load(config_path['run_model'])
    model_dir = Path(cfg['model_dir'])
    model = joblib.load(model_dir / 'music_model.pkl')

    context_size = cfg.get('context_size', 5)
    alpha = cfg.get('alpha', 0.5)
    sr = cfg.get('sample_rate', 16000)
    out_len = cfg.get('output_length', 100)
    n_mfcc = cfg.get('n_mfcc', 13)

    if cfg.get('seed') == 'zero':
        context = [np.zeros(n_mfcc) for _ in range(context_size)]
    else:
        context = [np.random.randn(n_mfcc) for _ in range(context_size)]

    generated = []

    for _ in range(out_len):
        weights = np.exp(-alpha * np.arange(context_size-1, -1, -1))
        weights = weights / weights.sum()
        context_vec = np.dot(weights, context[-context_size:])
        next_vec = model.predict(context_vec.reshape(1, -1))[0]
        generated.append(next_vec)
        context.append(next_vec)

    generated = np.stack(generated)  
    mel_spec = librosa.feature.inverse.mfcc_to_mel(generated.T)
    S = librosa.feature.inverse.mel_to_stft(mel_spec)
    y_out = librosa.griffinlim(S)

    wav_path = model_dir / 'generated_music.wav'
    wav_write(wav_path, sr, (y_out * 32767).astype(np.int16))
    print(f"Saved generated audio to {wav_path}")


def load_embedding(emb_type, vocab):
    if emb_type == "gpt2":
        return load_gpt2_embeddings(vocab)
    elif emb_type == "minilm":
        return load_minilm_embeddings(vocab, cache_path="cached_minilm_embeddings.npy")
    else:
        return load_glove_embeddings("glove.6B.50d.txt", vocab, embed_dim=50)

def train_model_collab(
    model_type,
    dataset_name,
    model_params,
    nested_model_params,
    embedding_type=None):

    model_dir = '.'
    data_dir  = '.'

    texts = load_texts({"dataset_name": dataset_name, 'sample_size': 6})
    train_texts, test_texts = train_test_split(texts, test_size=0.2, random_state=42)

    vocab = build_vocab(train_texts)
    ardt = ARDTEstimator(
        word2idx=vocab,
        embed_dim=50,
        embeddings=load_embedding(embedding_type, vocab),
        model_type=model_type,
        alpha=model_params.get('alpha', 0.47),
        context_size=model_params.get('context_size', 97),
        model_params=nested_model_params,
    )


    def build_xy(list_texts):
        ds = ARDTSimpleDataset(list_texts, vocab, ardt.embeddings,
                                context_size=ardt.context_size, alpha=ardt.alpha)
        X = np.vstack([x for x, _ in ds])
        y = np.vstack([ardt.embeddings[y] for _, y in ds])
        return X, y, ds

    X_train, y_train, train_ds = build_xy(train_texts)
    X_test,  y_true,  test_ds  = build_xy(test_texts)

    ardt.fit(train_texts)

    mean_cos_sim = ardt.score(X_test, y_true)
    #std_cos_sim = np.std(cos_sims)

  #  joblib.dump(ardt.model, model_dir / 'ardt_model.pkl')
    # save_yaml({
    #     'model_type': model_type,
    #     'context_size': ardt.context_size,
    #     'alpha': ardt.alpha,
    #     'embed_dim': ardt.embed_dim
    # }, model_dir / 'ardt_meta.yaml')
   # pickle.dump(train_ds, open(data_dir / 'ardt_dataset.pkl', 'wb'))

    # json.dump({
    #     'mean_cosine_similarity': mean_cos_sim,
    #    # 'std_cosine_similarity': std_cos_sim,
    #     'train_size': len(train_ds),
    #     'test_size': len(test_ds)
    # }, open(model_dir / 'results.json', 'w'), indent=2)

    print(f"[INFO] Model saved to {model_dir}")
    print(f"[RESULT] Cosine Similarity: mean = {mean_cos_sim:.4f}")



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="ARDT: train or run your model")
    subparsers = parser.add_subparsers(dest='command', required=True)

    tune_parser = subparsers.add_parser('tune', help='Tune selected ARDT model')
    tune_parser.add_argument('-c', '--config', default='config.yaml', help='Path to config YAML')

    train_parser = subparsers.add_parser('train', help='Train a new ARDT model')
    train_parser.add_argument('-c', '--config', default='config.yaml', help='Path to config YAML')

    run_parser = subparsers.add_parser('run', help='Run interactive ARDT generation')
    run_parser.add_argument('-c', '--config', default='config.yaml', help='Path to config YAML')

    music_train_parser = subparsers.add_parser('train_music', help='Train ARDT for music generation')
    music_train_parser.add_argument('-c', '--config', default='music_config.yaml', help='Path to music config YAML')
    
    music_run_parser = subparsers.add_parser('run_music', help='Run ARDT for music generation')
    music_run_parser.add_argument('-c', '--config', default='music_config.yaml', help='Path to music config YAML')
    
    wave_train_parser = subparsers.add_parser('train_wave', help='Train ARDT for wave generation')
    wave_train_parser.add_argument('-c', '--config', default='wave_config.yaml', help='Path to music config YAML')
    
    wave_run_parser = subparsers.add_parser('run_wave', help='Run ARDT for wave generation')
    wave_run_parser.add_argument('-c', '--config', default='wave_config.yaml', help='Path to music config YAML')

    args = parser.parse_args()

    train_model_collab(
        'lightgbm', 
        "roneneldan/TinyStories", 
        {'context_size': 97, 'alpha': 0.47},
        {
            # 'n_estimators': 100,
            # 'max_depth': 5,
            # 'learning_rate': 0.01,
            # 'n_jobs': -1,
            # "max_depth": 10,
            # "min_samples_split": 8,
            # "max_depth": 13,
            # "min_samples_leaf": 12,
            # "min_samples_split": 5
            # "n_jobs": -1,
            # "max_depth": 17,
            # "min_samples_leaf": 11,
            # "min_samples_split": 9,
            # "n_estimators": 198
        }, 
        'glove')

    # if args.command == 'train':
    #     train_model(args.config)
    # elif args.command == 'tune':
    #     nested_train_model(args.config)
    # elif args.command == 'run':
    #     run_model(args.config)
    # elif args.command == 'train_music':
    #     train_music_model_with_ardt(args.config)
    # elif args.command == 'run_music':
    #     run_music_model(args.config)
    # elif args.command == 'train_wave':
    #     from src.wav_facade.wave import train_wave_model_from_hf
    #     train_wave_model_from_hf(args.config)
    # elif args.command == 'run_wave':
    #     from src.wav_facade.wave import run_wavelet_model
    #     run_wavelet_model(args.config)
