"""
Preprocess LIVE_KT pivot dataset up to time T and export in pykt-compatible format,
preserving the same train/test user split as live_kt.compute_dataset_up_to.
"""

import os
import pandas as pd
import numpy as np


def dataset_up_to_T(df_pivot, T, i_train, i_test, skill=False, pad_val=-1, kfold=5):
    """
    Build a dataframe with one row per user containing only interactions from 1 to T.
    Uses the same train/test split (i_train, i_test) as live_kt for consistent evaluation.

    Args:
        df_pivot: DataFrame from live_kt.prepare_dataset (columns problem_id_1..N, correct_1..N,
                  optionally skill_id_1..N; index 0..n-1, column user_id).
        T: Last timestep (inclusive). Only interactions at positions 1..T are included.
        i_train: Index array of training users (from compute_dataset_up_to).
        i_test: Index array of test users (from compute_dataset_up_to).
        skill: If True, include concept/skill sequence (skill_id_*).
        pad_val: Value used for padding missing positions (pykt uses -1).
        kfold: Number of folds for train (pykt uses folds 0..kfold-1 for train/valid split).
               Train users are assigned fold 0..kfold-1 so the loader can use one fold for
               valid and the rest for train.

    Returns:
        DataFrame with columns: uid, fold, questions, concepts, responses.
        - fold: 0..kfold-1 for train, -1 for test (pykt convention).
        - questions/concepts/responses: comma-separated IDs/correct (no trailing padding here;
          pykt generate_sequences will pad to maxlen).
    """
    i_test_set = set(i_test)
    i_train_list = np.asarray([idx for idx in df_pivot.index if idx not in i_test_set])
    # Assign fold 0..kfold-1 to train users (deterministic by position) so pykt loader gets train/valid
    train_fold_by_idx = {}
    for pos, idx in enumerate(i_train_list):
        train_fold_by_idx[idx] = pos % kfold
    rows = []
    for idx in df_pivot.index:
        if idx in i_test_set:
            fold = -1
        else:
            fold = train_fold_by_idx[idx]
        uid = df_pivot.loc[idx, "user_id"]
        qs, cs, rs = [], [], []
        for i in range(1, T + 1):
            pid_col = f"problem_id_{i}"
            cor_col = f"correct_{i}"
            if pid_col not in df_pivot.columns:
                break
            pid = df_pivot.loc[idx, pid_col]
            cor = df_pivot.loc[idx, cor_col]
            if pd.isna(pid):
                continue  # trim: only include valid positions
            qs.append(str(int(pid)))
            if skill and f"skill_id_{i}" in df_pivot.columns:
                sk = df_pivot.loc[idx, f"skill_id_{i}"]
                cs.append(str(int(sk)) if pd.notna(sk) else str(int(pid)))
            else:
                cs.append(str(int(pid)))
            rs.append(str(int(cor)) if pd.notna(cor) else "0")
        if len(qs) == 0:
            continue
        rows.append({
            "uid": uid,
            "fold": fold,
            "questions": ",".join(qs),
            "concepts": ",".join(cs),
            "responses": ",".join(rs),
        })
    return pd.DataFrame(rows)


def compute_dataset_up_to_T(df_pivot, T, skill=False, max_context_size=200, right_align=True, kfold=5):
    """
    Compute the dataset up to timestep T and return the same train/test split
    as live_kt.compute_dataset_up_to, plus a pykt-ready total dataframe.

    Uses live_kt's logic (same train_test_split random_state=42) so that
    i_train and i_test match exactly for fair comparison.

    Args:
        df_pivot: From live_kt.prepare_dataset.
        T: Cutoff timestep (interactions 1..T only).
        skill: Whether skill/concept columns are present.
        max_context_size: Passed to compute_dataset_up_to (for compatibility).
        right_align: Passed to compute_dataset_up_to (unused in output here).
        kfold: Number of folds for train (train users get fold 0..kfold-1).

    Returns:
        total_df: DataFrame with uid, fold, questions, concepts, responses (pykt format).
        i_train: Training user row indices.
        i_test: Test user row indices.
    """
    from live_kt import compute_dataset_up_to

    _, _, i_train, i_test = compute_dataset_up_to(
        df_pivot, T, max_context_size=max_context_size,
        skill=skill, right_align=right_align,
    )
    total_df = dataset_up_to_T(df_pivot, T, i_train, i_test, skill=skill, kfold=kfold)
    return total_df, i_train, i_test


def to_pykt_dataset_up_to_T(
    df_pivot,
    T,
    dname,
    dataset_name,
    configf,
    skill=False,
    min_seq_len=3,
    maxlen=200,
    kfold=5,
):
    """
    Build dataset up to T with the same train/test interactions as live_kt,
    and write pykt-format files (train_valid_sequences.csv, test_sequences.csv, etc.) to dname.

    Args:
        df_pivot: From live_kt.prepare_dataset.
        T: Last timestep (inclusive).
        dname: Output directory for pykt files (created if needed).
        dataset_name: Dataset name for config.
        configf: Path to data_config.json.
        skill: Whether pivot has skill_id columns.
        min_seq_len: Min sequence length (pykt).
        maxlen: Max sequence length (pykt).
        kfold: Number of folds for train (pykt).

    Returns:
        total_df: The combined train+test dataframe in pykt row format.
        i_train: Training user indices (into original pivot).
        i_test: Test user indices (into original pivot).
    """
    from pykt.preprocess.split_datasets import main_from_dataframe

    total_df, i_train, i_test = compute_dataset_up_to_T(df_pivot, T, skill=skill, kfold=kfold)
    effective_keys = {"uid", "questions", "concepts", "responses"}
    main_from_dataframe(
        total_df,
        dname,
        dataset_name,
        configf,
        effective_keys,
        min_seq_len=min_seq_len,
        maxlen=maxlen,
        kfold=kfold,
    )
    return total_df, i_train, i_test
