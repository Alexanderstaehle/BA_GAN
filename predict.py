import itertools
import os
from pathlib import Path

import folium
import joblib
import s2sphere
import tensorflow as tf

from lstmwgan_2 import WGANGP

latent_dim = 100
n_epochs = 3
if __name__ == '__main__':
    random_latent_vectors = tf.random.normal(shape=(5, latent_dim))
    gan = WGANGP()
    gan.generator.load_weights('parameters/G_model_' + str(n_epochs) + '.h5')  # params/G_model_2000.h5
    generated_trajectories = gan.generator(random_latent_vectors).numpy()
    generated_trajectories = (generated_trajectories * 127.5) + 127.5
    for x in range(len(generated_trajectories)):
        scaler = joblib.load("preprocessing/scaler.pkl")
        generated_trajectories[x] = scaler.inverse_transform(generated_trajectories[x])
    s_c_id = list(itertools.chain(*generated_trajectories))
    cellId = []
    for i in range(0, len(s_c_id)):
        cellId.append(s_c_id[i][0])
    cellId = list(map(int, cellId))
    map_lat = []
    map_lng = []
    for i in range(0, len(s_c_id)):
        ll = str(s2sphere.CellId(cellId[i]).to_lat_lng())
        latlng = ll.split(',', 1)
        lat = latlng[0].split(':', 1)
        map_lat.append(float(lat[1]))
        map_lng.append(float(latlng[1]))
    points = list(zip(map_lat, map_lng))
    my_map = folium.Map(location=[46.3615142, 6.399388], zoom_start=10)
    folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(my_map)
    my_map.save(f"MDC_gen.html")
    print("Plot created")
