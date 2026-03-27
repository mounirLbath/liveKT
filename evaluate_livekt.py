from live_kt import *
import pandas as pd

# DATA_PATH = '../pykt-toolkit/data/assist2009/skill_builder_data_corrected_collapsed.csv'
DATA_PATH = '../pykt-toolkit/data/codeforces24/CF-data.csv'

def test_livekt():
    """
    Prepares Assistment 2009 dataset and runs Live KT on it
    """
    print(DATA_PATH)
    data = pd.read_csv(DATA_PATH)
    df = prepare_dataset(data, skill = True)
    test_models(df, skill = True, right_align = True)

def test_traditional_kt():
    """
    Prepares Assistment 2009 dataset and runs Live KT on PyKT models (AKT or DKT)
    """
    import os, sys, json

    from live_kt import prepare_dataset
    from preprocess_from_livekt import to_pykt_dataset_up_to_T
    
    # Set T here
    T = 20

    #Set model here ("dkt" for DKT)
    model = "dkt"
    print(DATA_PATH, T, model)

    # Prepare the same dataset to be run on a Pykt model
    ROOT = os.path.abspath(os.curdir)
    EXAMPLES, CONFIGS, DATA = os.path.join(ROOT, "examples"), os.path.join(ROOT, "configs"), os.path.join(ROOT, "data")
    dpath = os.path.join(DATA, "assist2009_livekt")
    ds_name = "assist2009_livekt"

    # path = os.path.join(DATA, "assist2009", "skill_builder_data_corrected_collapsed.csv")
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    if "order_id" in raw.columns: raw = raw.sort_values(["user_id", "order_id"])
    raw = raw[["user_id", "problem_id", "skill_id", "correct"]].copy(); raw["correct"] = raw["correct"].astype(int)
    df_pivot = prepare_dataset(raw, skill=True)

    to_pykt_dataset_up_to_T(df_pivot, T, dname=dpath, dataset_name=ds_name, configf=os.path.join(CONFIGS, "data_config.json"), skill=True, maxlen=200, kfold=5)
    with open(os.path.join(CONFIGS, "data_config.json")) as f: cfg = json.load(f)
    if ds_name not in cfg:
        cfg[ds_name] = {**cfg["assist2009_livekt"]}
    cfg[ds_name]["dpath"] = os.path.relpath(os.path.abspath(dpath), start=EXAMPLES)
    with open(os.path.join(CONFIGS, "data_config.json"), "w") as f: json.dump(cfg, f, indent=4)

    # Run Pykt model training and valid and display result
    import glob
    for p in glob.glob(os.path.join(dpath, "*.pkl")): os.remove(p)

    with open(os.path.join(CONFIGS, "kt_config.json")) as f: kt = json.load(f)
    params = {"dataset_name": ds_name, "model_name": model, "emb_type": "qid", "save_dir": os.path.join(ROOT, "saved_model"), "seed": 3407, "fold": 0, "use_wandb": 0, "add_uuid": 0}
    params.update(kt.get(model, {}))

    sys.path.insert(0, EXAMPLES)
    prev = os.getcwd(); os.chdir(EXAMPLES)
    from wandb_train import main
    import time
    t0 = time.time()
    main(params)
    total_sec = time.time() - t0
    print(f"Total training time: {total_sec:.1f}s ({total_sec/60:.1f} min)")
    os.chdir(prev); sys.path.pop(0)