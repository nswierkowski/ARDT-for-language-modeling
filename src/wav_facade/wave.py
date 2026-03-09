import numpy as np
from pathlib import Path
import yaml
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import pywt
from datasets import load_dataset

import numpy as np
from pathlib import Path
import yaml
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pywt
import wfdb
import pandas as pd

def train_wave_model_from_ptbxl(config_path):
    cfg = yaml.safe_load(open(config_path))['train_model']
    model_dir = Path(cfg['model_dir']); model_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(cfg['data_dir'])
    records_path = data_dir / 'records100'
    metadata_path = data_dir / 'ptbxl_database.csv'

    test_size = cfg.get('test_size', 0.2)
    wavelet = cfg.get('wavelet', 'db4')
    level = cfg.get('decomp_level', 3)

    print(f"[INFO] Loading metadata from: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata['filename_lr'].notnull()]

    print("[INFO] Extracting wavelet features...")
    X_list, y_list = [], []
    for _, row in metadata.iterrows():
        record_path = data_dir / row['filename_lr']
        try:
            record = wfdb.rdrecord(str(record_path))
            signal = record.p_signal[:, 0]  
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            features = np.concatenate(coeffs)
            X_list.append(features)
            y_list.append(row['scp_codes'])
        except Exception as e:
            print(f"[WARNING] Failed to process {record_path}: {e}")
            continue

    X = np.vstack(X_list)
    y = y_list

    le = LabelEncoder()
    y_encoded = le.fit_transform([list(eval(label).keys())[0] if label != '{}' else 'NORM' for label in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)

    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)

    joblib.dump(model, model_dir / 'wave_model.pkl')
    np.save(model_dir / 'X_train.npy', X_train)
    np.save(model_dir / 'X_test.npy', X_test)
    np.save(model_dir / 'y_train.npy', y_train)
    np.save(model_dir / 'y_test.npy', y_test)

    score = model.score(X_test, y_test)
    print(f"[RESULT] Test R² score: {score:.4f}")

    with open(model_dir / 'config_used.yaml', 'w') as f:
        yaml.dump(cfg, f)

    print(f"[INFO] Model saved to {model_dir}")

def extract_wavelet_features(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return np.concatenate(coeffs)

def train_wave_model_from_hf(config_path):
    cfg = yaml.safe_load(open(config_path))['train_model']
    model_dir = Path(cfg['model_dir']); model_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg.get('dataset_name', 'timeseries/ECG200')
    split = cfg.get('split', 'train')
    test_size = cfg.get('test_size', 0.2)
    wavelet = cfg.get('wavelet', 'db4')
    level = cfg.get('decomp_level', 3)

    print(f"[INFO] Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)

    print("[INFO] Extracting wavelet features...")
    X_list, y_list = [], []
    for sample in dataset:
        signal = np.array(sample['ts'])
        label = sample['label']

        coeffs = pywt.wavedec(signal, wavelet, level=level)
        features = np.concatenate(coeffs)
        X_list.append(features)
        y_list.append(label)

    X = np.vstack(X_list)
    y = np.array(y_list)

    if len(np.unique(y)) > 2:
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)

    joblib.dump(model, model_dir / 'wave_model.pkl')
    np.save(model_dir / 'X_train.npy', X_train)
    np.save(model_dir / 'X_test.npy', X_test)
    np.save(model_dir / 'y_train.npy', y_train)
    np.save(model_dir / 'y_test.npy', y_test)

    score = model.score(X_test, y_test)
    print(f"[RESULT] Test R² score: {score:.4f}")

    with open(model_dir / 'config_used.yaml', 'w') as f:
        yaml.dump(cfg, f)

    print(f"[INFO] Model saved to {model_dir}")

def run_wavelet_model(config_path):
    cfg = yaml.safe_load(open(config_path))['run_model']
    model_dir = Path(cfg['model_dir'])
    model = joblib.load(model_dir / 'wavelet_model.pkl')

    input_signal = np.load(cfg['input_signal'])
    features = extract_wavelet_features(input_signal)
    pred = model.predict(features.reshape(1, -1))
    
    print("[RESULT] Predicted output:", pred)
    np.save(model_dir / 'prediction.npy', pred)
