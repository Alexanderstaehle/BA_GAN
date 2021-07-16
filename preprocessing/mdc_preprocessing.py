import csv
import os
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import s2sphere
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler

import utils

max_length = 144


# convert an array of values into a dataset matrix
def create_dataset(data, look_back=1):
    data_x, data_y = [], []
    for j in range(len(data) - look_back - 1):
        a = data[j:(j + look_back), 0]
        data_x.append(a)
        data_y.append(data[j + look_back, 0])
    return np.array(data_x), np.array(data_y)


with open('../data/xyz.csv', 'r', ) as f:
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
    joblib.dump(scale_factor, 'scale_factor.pkl')

    timestamps = df['Time']

    borderbox = utils.fetchGeoLocation('Geneva, Suisse')
    # df = utils.dropOutlyingData(df, borderbox)
    points = list(zip(df.Latitude, df.Longitude))
    my_map = folium.Map(location=[46.3615142, 6.399388], zoom_start=10)
    borderbox = np.array(borderbox).astype(np.float)
    borderbox = [(borderbox[0], borderbox[2]), (borderbox[1], borderbox[3]), (borderbox[0], borderbox[3]),
                 (borderbox[1], borderbox[2])]
    folium.Rectangle(borderbox, color="blue", opacity=0.5).add_to(my_map)
    folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(my_map)
    out_path = os.path.join(Path(__file__).parent, 'plots', 'MDC')
    my_map.save(f"{out_path}.html")
    print("Plot created")

    r = s2sphere.RegionCoverer()

    dt = []

    for i in range(0, len(df['Latitude'])):
        p = s2sphere.LatLng.from_degrees(float(df['Latitude'][i]), float(df['Longitude'][i]))
        c = s2sphere.CellId.from_lat_lng(p)
        dt.append(c.id())

    dataset = []
    for row in dt:
        dataset.append(row)

    df = pd.DataFrame(dataset)
    dataset = df.values
    dataset = dataset.astype('float64')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    scaler_filename = "scaler.pkl"
    joblib.dump(scaler, scaler_filename)

    # TODO: Hier fehlt eine Unterteilung in einzelne Trajectories anstatt diesem Workaround
    # dataset = np.array(np.array_split(dataset, 675), dtype=object)
    dataset = np.array([dataset[i:i + np.random.randint(low=120, high=140)] for i in range(0, len(dataset))], dtype=object)
    dataset = dataset[..., np.newaxis]

    dataset = [pad_sequences(f, max_length, padding='pre', dtype='float64') for f in dataset]
    # reshape input to be [samples, time steps, features]
    dataset = np.reshape(dataset, (len(dataset), max_length, 1))

    # Rescale -1 to 1
    dataset = (dataset.astype(np.float64) - 127.5) / 127.5

    # timestamps = scaler.fit_transform(timestamps[0:1000])
    # dataset = np.c_[dataset, timestamps]

    # split into train and test sets
    train_size = int(len(dataset) * 0.85)
    test_size = int(len(dataset) * 0.10)
    validation_size = len(dataset) - train_size - test_size
    train, test, validation = dataset[0:train_size, :], dataset[train_size:len(dataset), :], dataset[
                                                                                             train_size + test_size:len(
                                                                                                 dataset), :]

    # reshape into X=t and Y=t+1
    # look_back = 1
    # trainX, trainY = create_dataset(train, look_back)
    # testX, testY = create_dataset(test, look_back)
    # validationX, validationY = create_dataset(validation, look_back)

    np.save('../data/preprocessed/train.npy', train)
    np.save('../data/preprocessed/test.npy', test)
    np.save('../data/preprocessed/validation.npy', validation)
