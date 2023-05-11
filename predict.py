import pandas as pd
import numpy as np
import pickle
import sys
import glob
import re
from sklearn.metrics import f1_score


def preprocess(path):
    data = np.array(np.zeros((1, 13)))
    cols = ''
    saved = 0
    ids = []

    # sort files in the given path
    source = path + '/*'
    files = sorted([(int(re.findall(r'\d+', s)[-1]), s) for s in glob.glob(source)])
    files = [s[1] for s in files]

    for f in files:

        df = pd.read_csv(f, sep='|')

        # Saving the file name
        name = f.split('/')[-1].split('.')[0]
        ids.append(name)

        # Drop columns
        df = df.drop(columns=df.iloc[:, 7:34].columns, axis=1)
        df = df.drop(['Unit1', 'Unit2'], axis=1)

        if not saved:
            cols = df.columns.tolist()
            saved = 1

        # If 'SepsisLabel' == 1 -> ignore rows
        indices = df[df['SepsisLabel'] == 1].index
        df = df.drop(indices[1:])
        label = 0 if len(indices) == 0 else 1

        # New feature
        hours = len(df)

        # Fill nulls
        df = df.interpolate(method='linear', limit_direction='both', axis=1)

        # Calculate mean values (labels column not included)
        row = df.iloc[:, :-1].mean().to_numpy()
        row = np.append(row, [hours, label])
        data = np.vstack([data, row])

    data = np.delete(data, 0, 0)  # Delete the first row
    cols.insert(11, 'Hours')
    data = pd.DataFrame(data, columns=cols)

    return data, ids


def main(argv):
    test_set, ids = preprocess(argv[1])
    X_test = test_set.iloc[:, :-1]
    y_test = test_set.iloc[:, -1]

    # Normalization
    scaler = pickle.load(open('./scaler.sav', 'rb'))
    X_test_scaled = scaler.transform(X_test)

    # Load the model and predict
    model = pickle.load(open('./xgb.sav', 'rb'))
    y_pred = model.predict(X_test_scaled)
    print(f'f1 score: {f1_score(y_test, y_pred)}')

    csv_df = pd.DataFrame({'id': ids, 'prediction': [int(y) for y in y_pred]})
    csv_df.to_csv('./prediction.csv', index=False)


if __name__ == '__main__':
    main(sys.argv)
