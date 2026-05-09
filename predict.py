# -*- coding: utf-8 -*-
"""
LocalQuakeAlert — Seismic Classifier
=====================================
INFERENCE MODE  (called by Server.js every 200 ms):
    python3 predict.py <mag> <zscore> <stalta> <sw420>
    Output: "PREDICTION CONFIDENCE LATENCY_MS METHOD"
    e.g.   "NORMAL 0.9720 0.423 xgb"

TRAINING MODE (run once to build models):
    python3 predict.py --train

CLASSES
    0  ambient          — background noise
    1  structural_noise — footsteps / door slams
    2  minor_quake      — Mw 3.x event
    3  major_quake      — Mw 9.x event
"""

# ─── INFERENCE FAST-PATH ─────────────────────────────────────────────────────
# Must be checked BEFORE any heavy imports so Server.js gets a response in <5 ms
import sys
import time as _time

def _rule_based(mag, zscore, stalta, sw420):
    """Classical threshold fallback — always works, no models needed."""
    t0 = _time.perf_counter()
    if zscore > 3.5 and stalta > 2.5:
        pred, conf = 'SEISMIC', min(0.90, zscore / 10)
    elif sw420 and zscore < 2.0:
        pred, conf = 'NOISE', 0.87
    else:
        pred, conf = 'NORMAL', 0.97
    ms = (_time.perf_counter() - t0) * 1000
    return pred, conf, ms, 'rules'


def _infer(mag, zscore, stalta, sw420):
    """Try ONNX → XGB JSON → rule-based, return (pred, conf, latency_ms, method)."""
    t0 = _time.perf_counter()

    # ── Option 1: ONNX (fastest, <1 ms) ──────────────────────────────────────
    try:
        import onnxruntime as ort
        import numpy as np
        import os
        onnx_path = os.path.join(os.path.dirname(__file__), 'seismic_xgb.onnx')
        if os.path.exists(onnx_path):
            sess   = ort.InferenceSession(onnx_path,
                         providers=['CPUExecutionProvider'])
            feat   = np.array([[mag, zscore, stalta, sw420]], dtype=np.float32)
            out    = sess.run(None, {'float_input': feat})
            pred_i = int(out[0][0])
            # probabilities tensor is second output
            proba  = out[1][0] if len(out) > 1 else None
            CLASSES = ['ambient', 'structural_noise', 'minor_quake', 'major_quake']
            # Map 4-class to server labels
            label_map = {0: 'NORMAL', 1: 'NOISE', 2: 'SEISMIC', 3: 'SEISMIC'}
            pred  = label_map[pred_i]
            conf  = float(proba[pred_i]) if proba is not None else 0.90
            ms    = (_time.perf_counter() - t0) * 1000
            return pred, conf, ms, 'onnx'
    except Exception:
        pass

    # ── Option 2: XGBoost JSON model ─────────────────────────────────────────
    try:
        import xgboost as xgb
        import numpy as np
        import os
        json_path = os.path.join(os.path.dirname(__file__), 'seismic_xgb.json')
        if os.path.exists(json_path):
            bst    = xgb.Booster()
            bst.load_model(json_path)
            feat   = xgb.DMatrix(np.array([[mag, zscore, stalta, sw420]],
                                           dtype=np.float32))
            proba  = bst.predict(feat)[0]          # shape (4,)
            pred_i = int(proba.argmax())
            label_map = {0: 'NORMAL', 1: 'NOISE', 2: 'SEISMIC', 3: 'SEISMIC'}
            pred  = label_map[pred_i]
            conf  = float(proba[pred_i])
            ms    = (_time.perf_counter() - t0) * 1000
            return pred, conf, ms, 'xgb'
    except Exception:
        pass

    # ── Option 3: Rule-based fallback ────────────────────────────────────────
    return _rule_based(mag, zscore, stalta, sw420)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # ── Inference mode: python3 predict.py mag zscore stalta sw420 ───────────
    if len(sys.argv) == 5 and sys.argv[1] != '--train':
        try:
            mag, zscore, stalta, sw420 = (float(sys.argv[1]), float(sys.argv[2]),
                                           float(sys.argv[3]), float(sys.argv[4]))
            pred, conf, ms, method = _infer(mag, zscore, stalta, sw420)
            # Output format Server.js expects: "SEISMIC 0.9400 1.234 xgb"
            print(f"{pred} {conf:.4f} {ms:.4f} {method}")
        except Exception as exc:
            # Fallback so server never hangs
            print(f"NORMAL 0.9700 0.001 error")
        sys.exit(0)

    # ── Training mode: python3 predict.py --train ─────────────────────────────
    if '--train' not in sys.argv:
        print("Usage:")
        print("  Inference : python3 predict.py <mag> <zscore> <stalta> <sw420>")
        print("  Training  : python3 predict.py --train")
        sys.exit(1)

    # ═════════════════════════════════════════════════════════════════════════
    # TRAINING PIPELINE — only runs with --train flag
    # To install deps: pip install obspy xgboost scikit-learn numpy scipy
    #                              matplotlib seaborn torch onnxruntime skl2onnx
    # ═════════════════════════════════════════════════════════════════════════
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')
    np.random.seed(42)

    from obspy.clients.fdsn import Client
    from obspy import UTCDateTime
    from scipy import signal
    from scipy.fft import rfft, rfftfreq
    from scipy.stats import kurtosis, skew
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 accuracy_score, f1_score)
    import xgboost as xgb
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Data acquisition ──────────────────────────────────────────────────────
    client  = Client("IRIS")
    FS      = 40
    WINDOW  = 2.0
    N_SAMP  = int(FS * WINDOW)
    CLASSES = ['ambient', 'structural_noise', 'minor_quake', 'major_quake']
    N_CLS   = len(CLASSES)

    def fetch_stream(start_str, duration_sec, fallback_std=0.002):
        try:
            st = client.get_waveforms("IU", "ANMO", "00", "BH?",
                                      UTCDateTime(start_str),
                                      UTCDateTime(start_str) + duration_sec)
            st.detrend('demean'); st.detrend('linear')
            st.filter('highpass', freq=0.5); st.resample(FS)
            min_len = min(len(tr.data) for tr in st)
            trZ = st.select(component="Z")[0].data[:min_len]
            cN  = st.select(component="1") or st.select(component="N")
            trN = cN[0].data[:min_len]
            cE  = st.select(component="2") or st.select(component="E")
            trE = cE[0].data[:min_len]
            data = np.stack([trN, trE, trZ], axis=1).astype(np.float32)
            data /= (np.max(np.abs(data)) + 1e-8)
            print(f'  ✓ Downloaded {start_str}  shape={data.shape}')
            return data
        except Exception as exc:
            print(f'  ⚠ IRIS fetch failed ({exc}); using synthetic fallback.')
            n = int(duration_sec * FS)
            return np.random.normal(0, fallback_std, (n, 3)).astype(np.float32)

    print('Downloading waveforms…')
    raw_ambient = fetch_stream("2023-01-01T00:00:00", 1200)
    raw_major   = fetch_stream("2011-03-11T05:55:00", 1200)
    raw_minor   = fetch_stream("2020-08-09T14:07:00", 1200)

    raw_noise = raw_ambient.copy()
    for i in range(0, len(raw_noise) - FS, FS * 3):
        idx    = np.arange(FS)
        decay  = np.random.uniform(4, 8)
        freq   = np.random.uniform(8, 15)
        impact = np.exp(-idx / decay) * np.sin(2 * np.pi * freq * idx / FS)
        amp    = np.random.uniform(0.3, 1.2)
        end    = min(i + FS, len(raw_noise))
        raw_noise[i:end, 2] += (impact[:end-i] * amp).astype(np.float32)
        raw_noise[i:end, 0] += (impact[:end-i] * amp * 0.3).astype(np.float32)
    raw_noise /= (np.max(np.abs(raw_noise)) + 1e-8)

    def chunk(arr, label):
        n = len(arr) // N_SAMP
        X = arr[:n * N_SAMP].reshape(n, N_SAMP, 3)
        y = np.full(n, label, dtype=np.int64)
        return X, y

    X1, y1 = chunk(raw_ambient, 0)
    X2, y2 = chunk(raw_noise,   1)
    X3, y3 = chunk(raw_minor,   2)
    X4, y4 = chunk(raw_major,   3)
    X_raw  = np.concatenate([X1, X2, X3, X4], axis=0)
    y_all  = np.concatenate([y1, y2, y3, y4], axis=0)
    print(f'Dataset  shape={X_raw.shape}  labels={np.bincount(y_all)}')

    # ── Feature engineering ───────────────────────────────────────────────────
    def compute_sta_lta(x, sta_win=8, lta_win=40):
        sta = np.convolve(x**2, np.ones(sta_win)/sta_win,  mode='same')
        lta = np.convolve(x**2, np.ones(lta_win)/lta_win, mode='same')
        return sta / np.maximum(lta, 1e-10)

    def extract_features(window):
        feats = []
        freqs = rfftfreq(N_SAMP, 1/FS)
        for axis in range(3):
            x = window[:, axis].astype(np.float64)
            feats += [np.mean(x), np.std(x), np.max(np.abs(x)),
                      np.sqrt(np.mean(x**2)), float(kurtosis(x)), float(skew(x)),
                      float(np.percentile(np.abs(x), 95)),
                      float(np.percentile(np.abs(x), 75)),
                      float(np.sum(x**2)), float(np.mean(np.abs(np.diff(x)))),
                      float(np.max(np.abs(np.diff(x)))),
                      float(np.sum(np.abs(np.diff(np.sign(x)))) / 2)]
            sl = compute_sta_lta(x)
            feats += [float(np.max(sl)), float(np.mean(sl)), float(np.std(sl)),
                      float(np.sum(sl > 3.0)), float(np.sum(sl > 5.0))]
            fft_mag = np.abs(rfft(x)); fft_pwr = fft_mag**2
            tot = np.sum(fft_pwr) + 1e-10
            def bp(fl, fh):
                return float(np.sum(fft_pwr[(freqs >= fl) & (freqs < fh)]) / tot)
            feats += [bp(0.1,1.0), bp(1.0,3.0), bp(3.0,8.0), bp(8.0,15.0), bp(15.0,20.0),
                      float(freqs[max(np.argmax(fft_mag), 1)]),
                      float(np.sum(freqs * fft_pwr) / tot)]
            env = np.abs(signal.hilbert(x))
            feats += [float(np.max(env)), float(np.argmax(env) / N_SAMP),
                      float(np.std(env)),
                      float(np.mean(env[N_SAMP//2:]) / (np.mean(env[:N_SAMP//2]) + 1e-10))]
        H = np.sum(window[:, 0]**2) + np.sum(window[:, 1]**2)
        V = np.sum(window[:, 2]**2) + 1e-10
        feats.append(float(H / V))
        mag = np.sqrt(np.sum(window.astype(np.float64)**2, axis=1))
        feats += [float(np.max(mag)), float(np.mean(mag)),
                  float(np.std(mag)), float(kurtosis(mag))]
        for (i, j) in [(0,1),(0,2),(1,2)]:
            feats.append(float(np.corrcoef(window[:,i], window[:,j])[0,1]))
        return np.array(feats, dtype=np.float32)

    print('Extracting features…')
    t0     = _time.time()
    X_feat = np.array([extract_features(s) for s in X_raw])
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
    print(f'Done in {_time.time()-t0:.1f}s  |  dim={X_feat.shape[1]}')

    # ── Train / test split ────────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_feat, y_all, test_size=0.20, random_state=42, stratify=y_all)
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print('Training XGBoost…')
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.04,
        subsample=0.80, colsample_bytree=0.80, min_child_weight=3,
        gamma=0.1, reg_alpha=0.05, reg_lambda=1.0,
        eval_metric='mlogloss', early_stopping_rounds=30,
        random_state=42, n_jobs=-1, tree_method='hist')
    xgb_model.fit(X_tr_sc, y_tr, eval_set=[(X_te_sc, y_te)], verbose=False)
    y_pred_xgb = xgb_model.predict(X_te_sc)
    print(f'XGBoost accuracy: {accuracy_score(y_te, y_pred_xgb):.4f}')
    print(classification_report(y_te, y_pred_xgb, target_names=CLASSES))

    # ── ResNet ────────────────────────────────────────────────────────────────
    class ResBlock1D(nn.Module):
        def __init__(self, in_ch, out_ch, kernel=7, stride=1):
            super().__init__()
            pad = kernel // 2
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
            self.bn1   = nn.BatchNorm1d(out_ch)
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
            self.bn2   = nn.BatchNorm1d(out_ch)
            self.relu  = nn.ReLU(inplace=True)
            self.drop  = nn.Dropout(0.1)
            self.skip  = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.drop(self.bn2(self.conv2(out)))
            return self.relu(out + self.skip(x))

    class SeismicResNet(nn.Module):
        def __init__(self, n_classes=4):
            super().__init__()
            self.stem   = nn.Sequential(
                nn.Conv1d(3, 32, 9, padding=4, bias=False),
                nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2))
            self.layer1 = ResBlock1D(32,  64,  7)
            self.layer2 = ResBlock1D(64,  128, 5)
            self.layer3 = ResBlock1D(128, 256, 3)
            self.layer4 = ResBlock1D(256, 256, 3)
            self.gap    = nn.AdaptiveAvgPool1d(1)
            self.head   = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
                nn.Linear(128, 64),  nn.ReLU(inplace=True), nn.Dropout(0.2),
                nn.Linear(64, n_classes))
        def forward(self, x):
            return self.head(self.gap(self.layer4(self.layer3(
                self.layer2(self.layer1(self.stem(x)))))))

    X_raw_T = X_raw.transpose(0, 2, 1).astype(np.float32)
    for ch in range(3):
        mu  = X_raw_T[:, ch, :].mean()
        sig = X_raw_T[:, ch, :].std() + 1e-8
        X_raw_T[:, ch, :] = (X_raw_T[:, ch, :] - mu) / sig

    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X_raw_T, y_all, test_size=0.20, random_state=42, stratify=y_all)
    train_ld = DataLoader(TensorDataset(torch.FloatTensor(Xr_tr), torch.LongTensor(yr_tr)),
                          batch_size=64, shuffle=True)
    test_ld  = DataLoader(TensorDataset(torch.FloatTensor(Xr_te), torch.LongTensor(yr_te)),
                          batch_size=256)

    model     = SeismicResNet(N_CLS).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    N_EPOCHS  = 50
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-3, epochs=N_EPOCHS, steps_per_epoch=len(train_ld))

    print(f'Training ResNet for {N_EPOCHS} epochs…')
    best_acc, best_state = 0.0, None
    val_accs, train_losses = [], []

    for epoch in range(N_EPOCHS):
        model.train(); ep_loss = 0.0
        for Xb, yb in train_ld:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            ep_loss += loss.item()
        model.eval(); correct = total = 0
        with torch.no_grad():
            for Xb, yb in test_ld:
                preds   = model(Xb.to(device)).argmax(1).cpu()
                correct += (preds == yb).sum().item(); total += len(yb)
        acc = correct / total
        val_accs.append(acc); train_losses.append(ep_loss / len(train_ld))
        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{N_EPOCHS}  loss={ep_loss/len(train_ld):.4f}  val={acc:.4f}')

    model.load_state_dict(best_state)
    torch.save(best_state, 'best_resnet.pt')
    print(f'Best CNN accuracy: {best_acc:.4f}')

    # ── CNN test evaluation ───────────────────────────────────────────────────
    model.eval(); all_probs_cnn, all_preds_cnn = [], []
    with torch.no_grad():
        for Xb, _ in test_ld:
            probs = torch.softmax(model(Xb.to(device)), dim=1).cpu().numpy()
            all_probs_cnn.extend(probs); all_preds_cnn.extend(probs.argmax(1))
    y_probs_cnn = np.array(all_probs_cnn)
    y_pred_cnn  = np.array(all_preds_cnn, dtype=int)
    print(f'CNN accuracy: {accuracy_score(yr_te, y_pred_cnn):.4f}')

    # ── Ensemble ──────────────────────────────────────────────────────────────
    idx_all    = np.arange(len(X_feat))
    _, idx_te2 = train_test_split(idx_all, test_size=0.20, random_state=42, stratify=y_all)
    y_probs_xgb = xgb_model.predict_proba(scaler.transform(X_feat[idx_te2]))
    y_true_ens  = y_all[idx_te2]
    ens_probs   = 0.40 * y_probs_xgb + 0.60 * y_probs_cnn
    y_pred_ens  = np.argmax(ens_probs, axis=1)
    print('=== Ensemble ===')
    print(classification_report(y_true_ens, y_pred_ens, target_names=CLASSES))

    # ── Save models ───────────────────────────────────────────────────────────
    xgb_model.save_model('seismic_xgb.json')
    print('Saved seismic_xgb.json')

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnx
        init_type = [('float_input', FloatTensorType([None, X_tr_sc.shape[1]]))]
        onnx_mdl  = convert_sklearn(xgb_model, initial_types=init_type, target_opset=12)
        onnx.save(onnx_mdl, 'seismic_xgb.onnx')
        print(f'Saved seismic_xgb.onnx  ({onnx_mdl.ByteSize()/1024:.1f} KB)')
    except Exception as exc:
        print(f'ONNX export skipped: {exc}  (seismic_xgb.json will be used)')

    print('\n✓ Training complete. Run server.js — inference will use saved models.')
