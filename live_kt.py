import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from skrub import tabular_pipeline
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier
import time
import os
import re, glob
import numpy as np

"""
Our implementation of LiveKT 
"""

# Models used to run liveKT
models = {'LR':LogisticRegression(solver='liblinear'),
          'GBM': tabular_pipeline('classifier'),
          'PFN': TabPFNClassifier(ignore_pretraining_limits=True),
          'ICL': TabICLClassifier()}

def prepare_dataset(data, skill = False, i_fold=0):
    """
    Prepare the dataset for evaluation

    - data: pandas dataframe with columns user_id, problem_id, skill_id, correct (int in {0,1})
            and sorted by time
    returns pandas dataframe of prepared dataset
    """
    df = data.copy()

    # Factorize IDs (optional but consistent with your problem_id approach)
    df['problem_id'], _ = pd.factorize(df['problem_id'])
    df['problem_id'] += 1

    if skill:
        df['skill_id'], _ = pd.factorize(df['skill_id'])
        df['skill_id'] += 1

    df['user_occ'] = df.groupby('user_id')['problem_id'].transform('count')
    df = df.query('user_occ > 5')

    # index sequences for each user
    df['attempt_nb'] = df.groupby('user_id').cumcount() + 1
    df['user'] = np.unique(df['user_id'], return_inverse=True)[1]

    # pivot dataset: now includes skill_id too
    if skill:
        df_pivot = df.pivot(index='user', columns='attempt_nb',
                        values=['problem_id', 'skill_id', 'correct'])
    else:
        df_pivot = df.pivot(index='user', columns='attempt_nb', values=['problem_id', 'correct'])

    df_pivot.columns = [f"{col}_{num}" for col, num in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    return df_pivot

def _right_align_row(row, start, t, prefix_list):
    """Right-align a single row: nan, nan, ..., v0, v1, ..., vk so last non-nan is at t.
    Alignment is driven by problem_id (same k/L/n_pad for all prefixes)."""
    out = row.copy()
    pid_cols = [f'problem_id_{i}' for i in range(start, t + 1)]
    pid_vals = row[pid_cols].values
    last = np.where(pd.notna(pid_vals))[0]
    if len(last) == 0:
        return out
    k = last[-1]
    L = k + 1
    n_cols = len(pid_cols)
    n_pad = n_cols - L
    for prefix in prefix_list:
        cols = [f'{prefix}_{i}' for i in range(start, t + 1)]
        vals = row[cols].values
        aligned = np.full(n_cols, np.nan, dtype=vals.dtype)
        aligned[n_pad:] = vals[:L]
        for i, c in enumerate(cols):
            out[c] = aligned[i]
    return out

def compute_dataset_up_to(df, t, max_context_size=200, skill = False, right_align = True, i_fold=0):
    start = max(1, t - max_context_size)

    if skill:
        df_slice = df[
            [f'problem_id_{i}' for i in range(start, t + 1)] +
            [f'skill_id_{i}'   for i in range(start, t + 1)] +
            [f'correct_{i}'    for i in range(start, t + 1)]
        ].copy()
    else:
        df_slice = df[
            [f'problem_id_{i}' for i in range(start, t + 1)] +
            [f'correct_{i}'    for i in range(start, t + 1)]
        ].copy()

    cv = ShuffleSplit(n_splits=5, random_state=42, test_size=0.2)
    i_train, i_test = list(cv.split(df.index))[i_fold]
    print(len(i_train), max(i_train), sum(i_train), len(i_test), max(i_test), sum(i_test))
    i_test = df_slice.iloc[i_test].dropna(subset=[f'correct_{t}']).index

    # Right-align train rows only: nan, nan, ..., i0, i1, ..., ik (k = last non-nan index)
    if right_align:
        prefix_list = ['problem_id', 'skill_id', 'correct'] if skill else ['problem_id', 'correct']
        for idx in i_train:
            if idx not in df_slice.index:
                continue
            df_slice.loc[idx] = _right_align_row(df_slice.loc[idx], start, t, prefix_list)

    X = df_slice.drop(columns=[f'correct_{t}'])      # keep problem_id_t and skill_id_t in X
    y = df_slice[f'correct_{t}'].fillna(0.0)

    print('Mean test outcome', y[i_test].mean())

    return X, y, i_train, i_test

def compute_results_pivot(model, model_name, df, t, results=None, results_time=None, fillna=False, skill = False, debug = True, right_align = True, i_fold=0):   
    """
    Computes the result 
    """
    X, y, i_train, i_test = compute_dataset_up_to(df,t, skill = skill, right_align = right_align, i_fold=i_fold)
    
    if len(i_test) == 0:
        return None
    
    if fillna:
        X = X.fillna(0.)
    t0 = time.perf_counter()

    # terminology can be misleading, there is no training or fitting in TabPFN as explained in our paper: 
    # for TabPFN, this "fit" only declares which are the train rows and which are the test rows, it does not actually "fit" anything
    model.fit(X.iloc[i_train], y.iloc[i_train])
    y_pred = model.predict_proba(X.iloc[i_test])[:, 1]

    inference_sec = time.perf_counter() - t0

    if not(results is None):
        score = roc_auc_score(y.iloc[i_test], y_pred)
        results.loc[model_name, t] = score
        if debug:
            print('Test AUC', score, 'on', len(i_test), 'samples at time',t)
    
    if not(results_time is None):
        results_time.loc[model_name, t] = inference_sec

    return (
        y.iloc[i_test],
        y_pred,
        i_test.to_numpy(), # row_idx
        np.full(len(i_test), t, dtype=np.int32), # t
        df.loc[i_test, f"problem_id_{t}"].to_numpy()
    )

def atomic_savez_compressed(path, **arrays):
    """Write .npz atomically (write temp then rename) to avoid partial files."""
    tmp_path = path[:-4] + ".tmp.npz" if path.endswith(".npz") else path + ".tmp.npz"
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)

def build_model_map(all_y, all_pred, all_row_idx, all_t, all_pid):
    return {
        "y_true": np.concatenate([np.asarray(x) for x in all_y]),
        "y_pred": np.concatenate([np.asarray(x) for x in all_pred]),
        "row_idx": np.concatenate([np.asarray(x) for x in all_row_idx]),
        "t": np.concatenate([np.asarray(x) for x in all_t]),
        "problem_id": np.concatenate([np.asarray(x) for x in all_pid]),
    }

def find_latest_npz(save_dir, model_name):
    """
    Returns the best resume file path for this model:
    - prefer final file MODEL.npz if it exists
    - else highest-t checkpoint MODEL_ckpt_tXXXX.npz
    - else None
    """
    final_path = os.path.join(save_dir, f"{model_name}.npz")
    if os.path.exists(final_path):
        return final_path

    ckpts = glob.glob(os.path.join(save_dir, f"{model_name}_ckpt_t*.npz"))
    if not ckpts:
        return None

    def get_t(p):
        m = re.search(r"_ckpt_t(\d+)\.npz$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    return max(ckpts, key=get_t)

def load_model_map_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def full_auc(df, start_t = 1, max_t = None,save_dir="predictions", model_list = ['LR', 'GBM', 'PFN', 'ICL'], checkpoint_every=100, save_checkpoints=True, resume=True):

    os.makedirs(save_dir, exist_ok=True)
    results = pd.DataFrame()

    for model_name in model_list:
        model = models[model_name]
        all_y, all_pred = [], []
        all_row_idx, all_t, all_pid = [], [], []

        i = start_t
        if resume:
            resume_path = find_latest_npz(save_dir, model_name)
            if resume_path is not None:
                old = load_model_map_npz(resume_path)
                all_y.append(old["y_true"])
                all_pred.append(old["y_pred"])
                all_row_idx.append(old["row_idx"])
                all_t.append(old["t"])
                all_pid.append(old["problem_id"])

                last_t = int(old["t"].max()) if old["t"].size else (start_t - 1)
                i = max(i, last_t + 1)
                print(f"Resuming {model_name} from {resume_path} (last_t={last_t})")


        print("---")
        print(f"Evaluating model {model_name}")

        while True and (max_t == None or i <= max_t):
            out = compute_results_pivot(model, model_name, df, i, fillna=(model_name != 'PFN'), debug=False)
            
            if out is None:
                break
            
            y, pred, row_idx, t_arr, pid = out

            all_y.append(y)
            all_pred.append(pred)
            all_row_idx.append(row_idx)
            all_t.append(t_arr)
            all_pid.append(pid)

            if save_checkpoints and checkpoint_every>0 and (i % checkpoint_every == 0):
                model_map = build_model_map(all_y, all_pred, all_row_idx, all_t, all_pid)
                ckpt_path = os.path.join(save_dir, f"{model_name}_ckpt_t{i:04d}.npz")
                atomic_savez_compressed(ckpt_path, **model_map)
                try:
                    ckpt_auc = roc_auc_score(model_map["y_true"], model_map["y_pred"])
                    print(f"Checkpoint t={i}: saved {ckpt_path} (running AUC={ckpt_auc:.6f})")
                except Exception:
                    print(f"Checkpoint t={i}: saved {ckpt_path}")

            i += 1

        # save predictions and probabilities
        model_map = build_model_map(all_y, all_pred, all_row_idx, all_t, all_pid)
        out_path = os.path.join(save_dir, f"{model_name}.npz")
        atomic_savez_compressed(out_path, **model_map)

        # compute AUC over whole dataset
        score = roc_auc_score(
            model_map["y_true"],
            model_map["y_pred"],
        )

        results.loc[model_name, "AUC"] = score
        print(f"Model {model_name} got total AUC of {score}")

    return results

def test_models(df, step=5, a=5, b=21, skill = False, right_align = True, i_fold=0):
    """
    Tests the different models on the LiveKT pipeline 
    returns results (AUC for each model for each T), and results_time (time to make prediction for each model and T)
    """
    results = pd.DataFrame()
    results_time = pd.DataFrame()
    results.columns.name = 't'

    for model_name, model in models.items():
        print("---")
        print("Evaluating model ",model_name)
        for i in range(a,b, step):
            compute_results_pivot(model, model_name, df, i, results, results_time, fillna=(model_name != 'PFN'), skill = skill, right_align = right_align, i_fold=i_fold)
    print(results_time)
    print(results)
    return results, results_time