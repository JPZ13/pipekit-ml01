import os
import pickle
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline


POS_EXAMPLE_DIR = '/tmp/aclImdb/train/pos/'
NEG_EXAMPLE_DIR = '/tmp/aclImdb/train/neg/'
SAVED_MODEL_LOCATION = '/tmp/model.pickle'


def _load_data(dir, max_files=4000):
    files = os.listdir(dir)
    result = []
    for f in files[:max_files]:
        with open(os.path.join(dir, f), 'r') as fp:
            result.append(fp.read())
    return result


if __name__ == '__main__':
    random.seed(523)
    np.random.seed(523)
    print('Loading positive examples...')
    pos_strings = _load_data(POS_EXAMPLE_DIR)
    print('Loading negative examples...')
    neg_strings = _load_data(NEG_EXAMPLE_DIR)

    random.shuffle(pos_strings)
    random.shuffle(neg_strings)

    num_pos = len(pos_strings)
    num_pos_train = int(.8 * num_pos)
    num_neg = len(neg_strings)
    num_neg_train = int(.8 * num_neg)

    X_train = pos_strings[:num_pos_train] + neg_strings[:num_neg_train]
    X_val = pos_strings[num_pos_train:] + neg_strings[num_neg_train:]
    y_train = np.append(np.ones(num_pos_train), np.zeros(num_neg_train))
    y_val = np.append(np.ones(num_pos - num_pos_train), np.zeros(num_neg - num_neg_train))

    pipeline = make_pipeline(
        CountVectorizer(max_features=1000),
        RandomForestClassifier()
    )

    print('Fitting model...')
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict_proba(X_val)[:, 1]
    print('AUC on validation set: {:.3f}'.format(roc_auc_score(y_val, y_val_pred)))

    with open(SAVED_MODEL_LOCATION, 'wb') as fp:
        pickle.dump(pipeline, fp)
