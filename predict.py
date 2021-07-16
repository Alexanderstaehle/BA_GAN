import folium
import joblib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from lstmwgan_trajgen import WGANGP
from preprocessing.utils import decodeTrajectories

latent_dim = 100
n_epochs = 60

if __name__ == '__main__':
    real_trajectories = np.load('data/preprocessed/train.npy', allow_pickle=True)
    real_trajectories = real_trajectories[:100]
    gen_inputs = []
    random_latent_vectors = noise = tf.random.normal((tf.shape(real_trajectories)[0], latent_dim), 0, 1)
    gen_inputs.append(real_trajectories)
    gen_inputs.append(random_latent_vectors)
    gan = WGANGP()
    gan.generator.load_weights('parameters/G_model4_' + str(n_epochs) + '.h5')  # params/G_model_2000.h5
    generated_trajectories = gan.generator(gen_inputs).numpy()
    scaler = joblib.load("preprocessing/scaler.pkl")
    gen_lat, gen_lng = decodeTrajectories(generated_trajectories, scaler)
    points = list(zip(gen_lat, gen_lng))
    my_map = folium.Map(location=[46.3615142, 6.399388], zoom_start=10)
    folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(my_map)
    my_map.save(f"MDC_gen.html")
    print("Plot created")

    xLabel = "Longitude"
    yLabel = "Latitude"
    real_lat, real_lng = decodeTrajectories(real_trajectories, scaler)
    heatmap, xedges, yedges = np.histogram2d(gen_lng, gen_lat, bins=20)
    heatmap2, xedges2, yedges2 = np.histogram2d(real_lng, real_lat, bins=20)

    # extent = [min(xedges[0], xedges2[0]), max(xedges[-1], xedges2[-1]), min(yedges[0], yedges2[0]),
    #          max(yedges[-1], yedges2[-1])]
    extent = [0, 50, 0, 50]
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    im1 = ax1.imshow(heatmap.T, extent=extent, origin='lower')
    im2 = ax2.imshow(heatmap2.T, extent=extent, origin='lower')
    plt.show()
