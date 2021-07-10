import itertools
import os
from pathlib import Path

import folium
import s2sphere
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from lstmwgan_2 import WGANGP

latent_dim = 100
n_epochs = 2
if __name__ == '__main__':
    random_latent_vectors = tf.random.normal(shape=(5, latent_dim))
    gan = WGANGP()
    gan.generator.load_weights('parameters/G_model_' + str(n_epochs) + '.h5')  # params/G_model_2000.h5
    generated_trajectories = gan.generator(random_latent_vectors)
    generated_trajectories = (generated_trajectories * 127.5) + 127.5
    scaler = MinMaxScaler(feature_range=(0, 1))
    generated_trajectories = scaler.inverse_transform(generated_trajectories)
    s_c_id = list(itertools.chain(*generated_trajectories))
    cellId = []
    for i in range(0, len(s_c_id)):
        cellId.append(s_c_id[i][0])
    cellId = map(int, cellId)
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
    out_path = os.path.join(Path(__file__).parent.parent, 'plots', 'MDC_gen')
    my_map.save(f"{out_path}.html")
    print("Plot created")
