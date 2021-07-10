from pathlib import Path
import numpy as np
import pandas as pd
import utils
import folium
import os

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

borderbox = utils.fetchGeoLocation('San Francisco, USA')
df = utils.dropOutlyingData(df, borderbox)
points = list(zip(df.Latitude, df.Longitude))
print("Preprocessing done")
my_map = folium.Map(location=[37.773972, -122.431297], zoom_start=10)
borderbox = np.array(borderbox).astype(np.float)
borderbox = [(borderbox[0], borderbox[2]), (borderbox[1], borderbox[3]), (borderbox[0], borderbox[3]), (borderbox[1], borderbox[2])]
folium.Rectangle(borderbox, color="blue", opacity=0.5).add_to(my_map)
folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(my_map)
out_path = os.path.join(Path(__file__).parent, 'plots', 'SF')
my_map.save(f"{out_path}.html")
print("Plot created")
