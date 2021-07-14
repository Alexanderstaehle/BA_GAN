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

user = '000'
dataDir = os.path.join(Path(__file__).parent.parent, 'data', 'Geolife Trajectories 1.3', 'Data')
gpsHeader = ["Latitude", "Longitude", "Zero", "Altitude", "Num of Days", "Date", "Time"]
outDir = os.path.join(Path(__file__).parent, 'plots')
if not (os.path.isdir(dataDir)):
    print(f"Data directory not found at: {dataDir}")
    exit()
boundingBox = utils.fetchGeoLocation("Beijing, China")

print(f"--- ON USER: {user} ---")

p = Path(outDir)
p2 = Path(os.path.join(outDir, user))

if not (os.path.isdir(p)):
    p.mkdir()


# Concat all user files
userPath = os.path.join(dataDir, user, 'Trajectory', '*')
allDirs = glob.glob(userPath)
my_map = folium.Map(location=[39.9075, 116.39723], zoom_start=14)
raw_data = pd.DataFrame(columns=gpsHeader)
for entry in allDirs:
    raw_data = pd.concat(
        [raw_data,
         pd.DataFrame(np.genfromtxt(entry, delimiter=',', skip_header=6, dtype='U'), columns=gpsHeader)])
raw_data[gpsHeader[0]] = pd.to_numeric(raw_data[gpsHeader[0]])
raw_data[gpsHeader[1]] = pd.to_numeric(raw_data[gpsHeader[1]])
raw_data[gpsHeader[2]] = pd.to_numeric(raw_data[gpsHeader[2]])
raw_data[gpsHeader[3]] = pd.to_numeric(raw_data[gpsHeader[3]])
raw_data[gpsHeader[4]] = pd.to_numeric(raw_data[gpsHeader[4]])

df = utils.dropOutlyingData(raw_data, boundingBox)
points = list(zip(df.Latitude, df.Longitude))
folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(my_map)
my_map.save(f"{p2}.html")
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
dataset = np.array(np.array_split(dataset, 1000), dtype=object)
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

np.save('../data/preprocessed/train_gl.npy', train)
np.save('../data/preprocessed/test_gl.npy', test)
np.save('../data/preprocessed/validation_gl.npy', validation)
print("Preprocessing done")
