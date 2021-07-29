import glob
import os
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import s2sphere
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import utils

max_length = 144

raw_data = []
map_lat = []
map_lng = []
map_time = []

dataDir = os.path.join(Path(__file__).parent.parent, 'data', 'Geolife Trajectories 1.3', 'Data')
gpsHeader = ["Latitude", "Longitude", "Zero", "Altitude", "Num of Days", "Date", "Time"]
outDir = os.path.join(Path(__file__).parent, 'plots')
if not (os.path.isdir(dataDir)):
    print(f"Data directory not found at: {dataDir}")
    exit()
borderbox = utils.fetchGeoLocation("Beijing, China")

my_map = folium.Map(location=[39.9075, 116.39723], zoom_start=14)

all_data = []
all_raw_data = []
r = s2sphere.RegionCoverer()
for i in range(0, 10):
    user = '00' + str(i)
    # get for each user
    print(f"--- ON USER: {user} ---")

    p = Path(outDir)
    p2 = Path(os.path.join(outDir, user))

    if not (os.path.isdir(p)):
        p.mkdir()
    # Get all user files
    userPath = os.path.join(dataDir, user, 'Trajectory', '*')
    allDirs = glob.glob(userPath)

    for entry in allDirs:
        raw_data = pd.DataFrame(columns=gpsHeader)
        raw_data = pd.concat(
            [raw_data,
             pd.DataFrame(np.genfromtxt(entry, delimiter=',', skip_header=6, dtype='U'), columns=gpsHeader)])
        raw_data[gpsHeader[0]] = pd.to_numeric(raw_data[gpsHeader[0]])
        raw_data[gpsHeader[1]] = pd.to_numeric(raw_data[gpsHeader[1]])
        # raw_data[gpsHeader[2]] = pd.to_numeric(raw_data[gpsHeader[2]])
        # raw_data[gpsHeader[3]] = pd.to_numeric(raw_data[gpsHeader[3]])
        # raw_data[gpsHeader[4]] = pd.to_numeric(raw_data[gpsHeader[4]])

        df = utils.dropOutlyingData(raw_data, borderbox)
        all_raw_data.append(df)

        dt = []

        for i in range(0, len(df['Latitude'])):
            p = s2sphere.LatLng.from_degrees(float(df['Latitude'][i]), float(df['Longitude'][i]))
            c = s2sphere.CellId.from_lat_lng(p)
            dt.append(c.id())

        all_data.append(dt)
for trajectory in all_raw_data:
    if len(trajectory) > 0:
        points = list(zip(trajectory.Latitude, trajectory.Longitude))
        folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(my_map)
borderbox = np.array(borderbox).astype(np.float)
borderbox = [(borderbox[0], borderbox[2]), (borderbox[1], borderbox[3]), (borderbox[0], borderbox[3]),
             (borderbox[1], borderbox[2])]
folium.Rectangle(borderbox, color="blue", opacity=0.5).add_to(my_map)
my_map.save(f"{p2}.html")
print("Plot created")
df = pd.concat(all_raw_data)

lat_centroid = df['Latitude'].sum() / len(df)
lon_centroid = df['Longitude'].sum() / len(df)
scale_factor = max(max(abs(df['Latitude'].max() - lat_centroid),
                       abs(df['Latitude'].min() - lat_centroid),
                       ),
                   max(abs(df['Longitude'].max() - lon_centroid),
                       abs(df['Longitude'].min() - lon_centroid),
                       ))
joblib.dump(scale_factor, 'scale_factor_gl.pkl')
dataset = [np.asarray(xi, dtype=object) for xi in all_data]

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
lengths = []
for i in range(0, len(dataset)):
    lengths.append(len(dataset[i]))
dataset = np.array(dataset, dtype=object)
# Flatten
dataset = np.hstack(dataset)
dataset = dataset[..., np.newaxis]
dataset = scaler.fit_transform(dataset)
scaler_filename = "scaler_gl.pkl"
joblib.dump(scaler, scaler_filename)

# Split again
dataset_split = []
for i in range(0, len(lengths)):
    dataset_split.append(np.array(dataset[:lengths[i]], dtype=object))
dataset_split = np.array(dataset_split, dtype=object)

# Padding
dataset = pad_sequences(dataset_split, max_length, padding='pre', dtype='float64')

# Rescale -1 to 1
dataset = (dataset.astype(np.float64) - 127.5) / 127.5

# split into train and test sets
train_size = int(len(dataset) * 0.85)
test_size = int(len(dataset) * 0.10)
validation_size = len(dataset) - train_size - test_size
train, test, validation = dataset[0:train_size, :], dataset[train_size:len(dataset), :], dataset[
                                                                                         train_size + test_size:len(
                                                                                             dataset), :]

np.save('../data/preprocessed/train_gl.npy', train)
np.save('../data/preprocessed/test_gl.npy', test)
np.save('../data/preprocessed/validation_gl.npy', validation)
print("Preprocessing done")
