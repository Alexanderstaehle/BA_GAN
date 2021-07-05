import csv
import os
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import s2sphere
from sklearn.preprocessing import MinMaxScaler

import utils


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

    timestamps = df['Time']

    borderbox = utils.fetchGeoLocation('Geneva, Suisse')
    # df = utils.dropOutlyingData(df, borderbox)
    points = list(zip(df.Latitude, df.Longitude))
    my_map = folium.Map(location=[46.3615142, 6.399388], zoom_start=10)
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
    dataset = dataset[0:1000]

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # timestamps = scaler.fit_transform(timestamps[0:1000])
    # dataset = np.c_[dataset, timestamps]

    # AB HIER DANN IN DER RICHTIGEN IMPLEMENTIERUNG
    # split into train and test sets
    train_size = int(len(dataset) * 0.85)
    test_size = int(len(dataset) * 0.10)
    validation_size = len(dataset) - train_size - test_size
    train, test, validation = dataset[0:train_size, :], dataset[train_size:len(dataset), :], dataset[
                                                                                             train_size + test_size:len(
                                                                                                 dataset), :]

    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    validationX, validationY = create_dataset(validation, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    validationX = np.reshape(validationX, (validationX.shape[0], validationX.shape[1], 1))

    np.save('../data/preprocessed/train.npy', trainX)
    np.save('../data/preprocessed/test.npy', testX)
    np.save('../data/preprocessed/validation.npy', validationX)
