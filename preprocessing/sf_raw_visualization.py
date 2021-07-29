import os
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import s2sphere
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler

import utils

max_length = 144
raw_data = []
with open("../data/new_abboip.txt", mode="r", encoding="utf-8") as f:
    for row in f:
        raw_data.append(row.split())
    np_data = np.array(raw_data)

df = pd.DataFrame(np_data, columns=["Latitude", "Longitude", "ID", "Time"])
df['Latitude'] = pd.to_numeric(df['Latitude'])
df['Longitude'] = pd.to_numeric(df['Longitude'])
df['ID'] = pd.to_numeric(df['ID'])
df['Time'] = pd.to_numeric(df['Time'])
df['Time'] = pd.to_datetime(df['Time'], unit='s')

borderbox = utils.fetchGeoLocation('San Francisco, USA')
df = utils.dropOutlyingData(df, borderbox)
points = list(zip(df.Latitude, df.Longitude))
my_map = folium.Map(location=[37.773972, -122.431297], zoom_start=10)
borderbox = np.array(borderbox).astype(np.float)
borderbox = [(borderbox[0], borderbox[2]), (borderbox[1], borderbox[3]), (borderbox[0], borderbox[3]),
             (borderbox[1], borderbox[2])]
folium.Rectangle(borderbox, color="blue", opacity=0.5).add_to(my_map)
folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(my_map)
out_path = os.path.join(Path(__file__).parent, 'plots', 'SF')
my_map.save(f"{out_path}.html")
print("Plot created")

lat_centroid = df['Latitude'].sum() / len(df)
lon_centroid = df['Longitude'].sum() / len(df)
scale_factor = max(max(abs(df['Latitude'].max() - lat_centroid),
                       abs(df['Latitude'].min() - lat_centroid),
                       ),
                   max(abs(df['Longitude'].max() - lon_centroid),
                       abs(df['Longitude'].min() - lon_centroid),
                       ))
joblib.dump(scale_factor, 'scale_factor_sf.pkl')
print("Preprocessing done")
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
# Vlt splitten in stündliche Trajectories
dataset = np.array([dataset[i:i + np.random.randint(low=120, high=140)] for i in range(0, len(dataset))], dtype=object)
dataset = dataset[..., np.newaxis]

dataset = [pad_sequences(f, max_length, padding='pre', dtype='float64') for f in dataset]
# reshape input to be [samples, time steps, features]
dataset = np.reshape(dataset, (len(dataset), max_length, 1))

# Rescale -1 to 1
dataset = (dataset.astype(np.float64) - 127.5) / 127.5

# split into train and test sets
train_size = int(len(dataset) * 0.85)
test_size = int(len(dataset) * 0.10)
validation_size = len(dataset) - train_size - test_size
train, test, validation = dataset[0:train_size, :], dataset[train_size:len(dataset), :], dataset[
                                                                                         train_size + test_size:len(
                                                                                             dataset), :]

np.save('../data/preprocessed/train_sf.npy', train)
np.save('../data/preprocessed/test_sf.npy', test)
np.save('../data/preprocessed/validation_sf.npy', validation)
