import numpy as np
import pandas as pd
import utils
import sys
from pathlib import Path
import folium
import glob
import os

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
boundingBox = utils.fetchGeoLocation("Beijing, China")
args = sys.argv

if len(args) <= 1:
    print(f"Usage: py3 gen_user_map.py {dataDir}/USER_ID ")
    print("Usage: See header of py file")
    exit()

path = args[1].split("/")
user = path[len(path) - 1]

print(f"--- ON USER: {user} ---")

p = Path(outDir)
p2 = Path(os.path.join(outDir, user))

if not (os.path.isdir(p)):
    p.mkdir()


# Concat all user_month files
userPath = os.path.join(dataDir, user, 'Trajectory', '*')
allDirs = glob.glob(userPath)
my_map = folium.Map(location=[39.9075, 116.39723], zoom_start=14)
raw_data = pd.DataFrame(columns=gpsHeader)
for entry in allDirs:
    # np.genfromtxt(entry, delimiter=',', skip_header=6, dtype='U')
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
print("Preprocessing done")
folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(my_map)
my_map.save(f"{p2}.html")
print("Plot created")
