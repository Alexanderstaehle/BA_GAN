import sys

import folium
import joblib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import polynomial_kernel
from tqdm import tqdm

from lstmwgan import WGANGP
from preprocessing.utils import decodeTrajectories

latent_dim = 100
n_epochs = 30
n_samples = 100

if __name__ == '__main__':
    real_trajectories = np.load('data/preprocessed/train_gl.npy', allow_pickle=True)
    real_trajectories = real_trajectories[:n_samples]
    gen_inputs = []
    random_latent_vectors = noise = tf.random.normal((tf.shape(real_trajectories)[0], latent_dim), 0, 1)
    gen_inputs.append(real_trajectories)
    gen_inputs.append(random_latent_vectors)
    gan = WGANGP()
    gan.generator.load_weights('parameters/G_model_gl_' + str(n_epochs) + '.h5')  # params/G_model_100.h5
    generated_trajectories = gan.generator(gen_inputs).numpy()
    scaler = joblib.load("preprocessing/scaler.pkl")
    gen_lat, gen_lng = decodeTrajectories(generated_trajectories, scaler)
    points = list(zip(gen_lat, gen_lng))
    # my_map = folium.Map(location=[46.3615142, 6.399388], zoom_start=10)
    my_map = folium.Map(location=[39.9075, 116.39723], zoom_start=14)
    folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(my_map)
    my_map.save(f"MDC_gen.html")
    print("Plot created")

    # -----------------HEATMAP-------------------
    xLabel = "Longitude"
    yLabel = "Latitude"
    real_lat, real_lng = decodeTrajectories(real_trajectories, scaler)
    heatmap, xedges, yedges = np.histogram2d(gen_lng, gen_lat, bins=20)
    heatmap2, xedges2, yedges2 = np.histogram2d(real_lng, real_lat, bins=20)

    extent = [0, 50, 0, 50]
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    im1 = ax1.imshow(heatmap.T, extent=extent, origin='lower')
    im2 = ax2.imshow(heatmap2.T, extent=extent, origin='lower')
    plt.show()

    # -----------------MMD-------------------
    """
    def compute_kernel(x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


    def compute_mmd(x, y):  # [batch_size, z_dim] [batch_size, z_dim]
        x_kernel = compute_kernel(x, x)
        y_kernel = compute_kernel(y, y)
        xy_kernel = compute_kernel(x, y)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


    real_trajectories = np.float32(real_trajectories)
    mmds = []
    for index in range(0, n_samples):
        mmd = compute_mmd(real_trajectories[index], generated_trajectories[index])
        mmds.append(mmd.numpy())
    mean_mmd = np.mean(mmds)
    print(mean_mmd)
"""


    def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000,
                                ret_var=True, output=sys.stdout, **kernel_args):
        m = min(codes_g.shape[0], codes_r.shape[0])
        mmds = np.zeros(n_subsets)
        if ret_var:
            vars = np.zeros(n_subsets)
        choice = np.random.choice

        with tqdm(range(n_subsets), desc='MMD', file=output) as bar:
            for i in bar:
                g = codes_g[choice(len(codes_g), subset_size, replace=False)]
                r = codes_r[choice(len(codes_r), subset_size, replace=False)]
                o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
                if ret_var:
                    mmds[i], vars[i] = o
                else:
                    mmds[i] = o
                bar.set_postfix({'mean': mmds[:i + 1].mean()})
        return (mmds, vars) if ret_var else mmds


    def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
                       var_at_m=None, ret_var=True):
        # use  k(x, y) = (gamma <x, y> + coef0)^degree
        # default gamma is 1 / dim
        X = codes_g
        Y = codes_r

        K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
        K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
        K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

        return _mmd2_and_variance(K_XX, K_XY, K_YY,
                                  var_at_m=var_at_m, ret_var=ret_var)


    def _sqn(arr):
        flat = np.ravel(arr)
        return flat.dot(flat)


    def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                           mmd_est='unbiased', block_size=1024,
                           var_at_m=None, ret_var=True):
        # based on
        # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
        # but changed to not compute the full kernel matrix at once
        m = K_XX.shape[0]
        assert K_XX.shape == (m, m)
        assert K_XY.shape == (m, m)
        assert K_YY.shape == (m, m)
        if var_at_m is None:
            var_at_m = m

        # Get the various sums of kernels that we'll use
        # Kts drop the diagonal, but we don't need to compute them explicitly
        if unit_diagonal:
            diag_X = diag_Y = 1
            sum_diag_X = sum_diag_Y = m
            sum_diag2_X = sum_diag2_Y = m
        else:
            diag_X = np.diagonal(K_XX)
            diag_Y = np.diagonal(K_YY)

            sum_diag_X = diag_X.sum()
            sum_diag_Y = diag_Y.sum()

            sum_diag2_X = _sqn(diag_X)
            sum_diag2_Y = _sqn(diag_Y)

        Kt_XX_sums = K_XX.sum(axis=1) - diag_X
        Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
        K_XY_sums_0 = K_XY.sum(axis=0)
        K_XY_sums_1 = K_XY.sum(axis=1)

        Kt_XX_sum = Kt_XX_sums.sum()
        Kt_YY_sum = Kt_YY_sums.sum()
        K_XY_sum = K_XY_sums_0.sum()

        if mmd_est == 'biased':
            mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                    + (Kt_YY_sum + sum_diag_Y) / (m * m)
                    - 2 * K_XY_sum / (m * m))
        else:
            assert mmd_est in {'unbiased', 'u-statistic'}
            mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
            if mmd_est == 'unbiased':
                mmd2 -= 2 * K_XY_sum / (m * m)
            else:
                mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m - 1))

        if not ret_var:
            return mmd2

        Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
        Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
        K_XY_2_sum = _sqn(K_XY)

        dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
        dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

        m1 = m - 1
        m2 = m - 2
        zeta1_est = (
                1 / (m * m1 * m2) * (
                _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
                - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
                + 1 / (m * m * m1) * (
                        _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
                - 2 / m ** 4 * K_XY_sum ** 2
                - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
                + 2 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
        )
        zeta2_est = (
                1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
                - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
                + 2 / (m * m) * K_XY_2_sum
                - 2 / m ** 4 * K_XY_sum ** 2
                - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
                + 4 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
        )
        var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
                   + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

        return mmd2, var_est


    output = {}
    generated_trajectories = np.reshape(generated_trajectories,
                                        [generated_trajectories.shape[0], generated_trajectories.shape[1]])
    real_trajectories = np.reshape(real_trajectories,
                                   [real_trajectories.shape[0], real_trajectories.shape[1]])
    ret = polynomial_mmd_averages(
        generated_trajectories, real_trajectories, degree=3, gamma=None,
        coef0=1, ret_var=False,
        n_subsets=100, subset_size=100)
    if False:
        output['mmd2'], output['mmd2_var'] = mmd2s, vars = ret
    else:
        output['mmd2'] = mmd2s = ret
    print("mean MMD^2 estimate:", mmd2s.mean())
    # print("std MMD^2 estimate:", mmd2s.std())
    # print("MMD^2 estimates:", mmd2s, sep='\n')
