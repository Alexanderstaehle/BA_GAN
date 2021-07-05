import csv
import sys

import numpy as np
import pandas as pd

from lstm_model import LSTM_TrajGAN

if __name__ == '__main__':
    n_epochs = int(sys.argv[1])
    n_batch_size = int(sys.argv[2])
    n_sample_interval = int(sys.argv[3])

    latent_dim = 100
    max_length = 144

    keys = ['cid', ]  # 'day', 'hour', 'category', 'mask']
    vocab_size = {"cid": 1, }  # "day": 7, "hour": 24, "category": 10, "mask": 1}

    with open('data/xyz.csv', 'r', ) as f:
        next(f)
        gps_logs = csv.reader(f)

        raw_logs = []

        for row in gps_logs:
            raw_logs.append(row)

        logs = np.array(raw_logs)

        df = pd.DataFrame(logs, columns=["Longitude", "Latitude", "Time"])
        df['Latitude'] = pd.to_numeric(df['Latitude'])
        df['Longitude'] = pd.to_numeric(df['Longitude'])
        df['Time'] = pd.to_numeric(df['Time'])

    lat_centroid = df['Latitude'].sum() / len(df)
    lon_centroid = df['Longitude'].sum() / len(df)

    scale_factor = max(max(abs(df['Latitude'].max() - lat_centroid),
                           abs(df['Latitude'].min() - lat_centroid),
                           ),
                       max(abs(df['Longitude'].max() - lon_centroid),
                           abs(df['Longitude'].min() - lon_centroid),
                           ))

    gan = LSTM_TrajGAN(latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, scale_factor)

    gan.train(epochs=n_epochs, batch_size=n_batch_size, sample_interval=n_sample_interval)
