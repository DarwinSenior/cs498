import pandas as pd
import numpy as np


def write_result(name, y):
    columns = ['Id', 'Prediction']
    values = np.zeros(y.size, dtype='int32, int32')
    values.dtype.names = columns
    df = pd.DataFrame(values, columns=columns)
    df['Id'] = np.arange(0, y.size)
    df['Prediction'] = y.astype(int)
    df.to_csv(name, sep=',', index=False)
