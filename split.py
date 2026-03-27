import pandas as pd
from evaluate_livekt import DATA_PATH
from live_kt import models
from sklearn.model_selection import cross_validate, GroupShuffleSplit, train_test_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np


df = pd.read_csv(DATA_PATH)
df['correct'] = df['correct'].astype(int)
df['user'] = np.unique(df['user_id'], return_inverse=True)[1]
df['item'] = np.unique(df['problem_id'], return_inverse=True)[1]
df['skill'] = np.unique(df['skill_id'].astype('string').fillna('0'), return_inverse=True)[1]
df['user_occ'] = df.groupby('user_id')['problem_id'].transform('count')
df['attempt_nb'] = df.groupby('user_id').cumcount()
# Remove too rare users and truncate at 20 samples per user
df = df.query('user_occ > 5 and attempt_nb < 20')

rng = np.random.default_rng(42)
N_USERS = df['user'].nunique()
selected_users = rng.choice(df['user'].unique(), N_USERS)
subset = df#.query('user in @selected_users')

FEATURES = ['user', 'item', 'skill', 'attempt_nb'] # was 'attempt_count' for Assistments
X = subset[FEATURES].fillna(0)  # Some skill_ids are NaN!
y = subset['correct']

cv = GroupShuffleSplit(n_splits=1, random_state=42, test_size=0.25)

"""
Some tests you can make
set(X.iloc[i_train]['user'].unique()) & set(X.iloc[i_test]['user'].unique())
set(X.iloc[i_train]['user'].unique()) & set(X.iloc[i_valtest]['user'].unique())
set(X.iloc[i_trainval]['user'].unique()) & set(X.iloc[i_test]['user'].unique())
"""

for i_train, i_valtest in cv.split(X, y, subset['user']):  # Currently only one split
    print(i_train.shape, i_valtest.shape)

    # i_val, i_test = train_test_split(i_valtest, test_size=0.5)
    mydata = X.iloc[i_valtest].copy()
    mydata['indice'] = i_valtest

    i_val = mydata.query('attempt_nb < 9')['indice']
    i_test = mydata.query('attempt_nb == 9')['indice']
    print(i_val.shape, i_test.shape)
    
    i_trainval = np.concatenate((i_train, i_val))
    print(i_trainval.shape)


def compute_auc(model, X, y, i_trainval, i_test):
    model.fit(X.iloc[i_trainval], y.iloc[i_trainval])
    y_pred = model.predict_proba(X.iloc[i_test])[:, 1]
    print('Test AUC', roc_auc_score(y.iloc[i_test], y_pred))


model = models['ICL']
compute_auc(model, X[['user', 'item', 'skill']], y, i_trainval, i_test)
compute_auc(model, X[['user', 'item', 'skill', 'attempt_nb']], y, i_trainval, i_test)
