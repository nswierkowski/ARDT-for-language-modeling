import argparse
import json
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import joblib
from src.models.TabularARDT import TabularARDTEstimator


def train_and_save(args):
    print(f"Loading dataset {args.dataset} split={args.split}...")
    ds = load_dataset(args.dataset, split=args.split)
    if args.n_rows:
        ds = ds.select(range(args.n_rows))

    print("Building vocabulary...")
    all_tokens = set(tok for seq in ds['midi_tokens'] for tok in seq)
    word2idx = {tok: i for i, tok in enumerate(sorted(all_tokens))}

    print("Encoding genres...")
    genres = sorted(set(ds['genre']))
    genre2idx = {g: i for i, g in enumerate(genres)}
    X_tab = np.zeros((len(ds), len(genres)), dtype=np.float32)
    for i, g in enumerate(ds['genre']):
        X_tab[i, genre2idx[g]] = 1.0

    sequences = ds['midi_tokens']

    print(f"Splitting data: test_size={args.test_size}...")
    X_tab_train, X_tab_test, seq_train, seq_test = train_test_split(
        X_tab, sequences, test_size=args.test_size,
        random_state=args.random_seed
    )

    model_params = json.loads(args.model_params)

    print("Initializing estimator...")
    estimator = TabularARDTEstimator(
        word2idx=word2idx,
        embed_dim=args.embed_dim,
        embeddings=None,
        context_size=args.context_size,
        alpha=args.alpha,
        mode='regression',
        model_type=args.model_type,
        model_params=model_params
    )
    estimator.genre2idx = genre2idx

    print("Fitting model...")
    estimator.fit(X_tab_train, seq_train)

    if args.save_path:
        print(f"Saving model to {args.save_path}...")
        joblib.dump(estimator, args.save_path)

    if args.preview:
        print("Previewing predictions:")
        preds = estimator.predict(X_tab_test, seq_test)
        for i in range(min(5, len(preds))):
            ctx = seq_test[i][-args.context_size:]
            print(f"Context: {ctx}\nPredicted: {preds[i]}\n")


def generate_from_saved(args):
    print(f"Loading model from {args.model_path}...")
    estimator = joblib.load(args.model_path)

    genre = args.genre
    if genre not in estimator.genre2idx:
        raise ValueError(f"Unknown genre '{genre}'. Available: {list(estimator.genre2idx.keys())}")
    one_hot = np.zeros((1, len(estimator.genre2idx)), dtype=np.float32)
    one_hot[0, estimator.genre2idx[genre]] = 1.0

    seq = []
    print(f"Generating {args.length} tokens for genre '{genre}'...")
    for _ in range(args.length):
        tok = estimator.predict(one_hot, [seq])[0]
        seq.append(tok)
    print("Generated sequence:", seq)


def main():
    parser = argparse.ArgumentParser(
        description="Train, save, or generate with TabularARDTEstimator on MidiCaps-preprocessed"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_p = subparsers.add_parser('train')
    train_p.add_argument('--dataset', type=str, default='benjamintli/MidiCaps-preprocessed')
    train_p.add_argument('--split', type=str, default='train')
    train_p.add_argument('--n_rows', type=int, default=None)
    train_p.add_argument('--context_size', type=int, default=16)
    train_p.add_argument('--alpha', type=float, default=0.5)
    train_p.add_argument('--embed_dim', type=int, default=128)
    train_p.add_argument('--model_type', choices=['decision_tree','random_forest','xgboost','lightgbm'], default='lightgbm')
    train_p.add_argument('--model_params', type=str, default='{}')
    train_p.add_argument('--test_size', type=float, default=0.2)
    train_p.add_argument('--random_seed', type=int, default=42)
    train_p.add_argument('--preview', action='store_true')
    train_p.add_argument('--save_path', type=str, default='tabular_ardt.pkl')

    gen_p = subparsers.add_parser('generate')
    gen_p.add_argument('--model_path', type=str, required=True)
    gen_p.add_argument('--genre', type=str, required=True)
    gen_p.add_argument('--length', type=int, default=50, help='Number of tokens to generate')

    args = parser.parse_args()
    if args.command == 'train':
        train_and_save(args)
    elif args.command == 'generate':
        generate_from_saved(args)


if __name__ == '__main__':
    main()
