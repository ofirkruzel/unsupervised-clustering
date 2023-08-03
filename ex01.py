import additional_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_classification , make_circles,make_s_curve
# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
from sklearn import datasets

sigma = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]  # Covariance matrices for each component
p = [0.5, 0.5]  # Probabilities for each component


def gaussian_data_2D_one_center():
    data, labels = make_blobs(n_samples=1000, centers=1, cluster_std=0.5, random_state=0, n_features=2)
    pred1, accuracy1=additional_function.gauss_mle_cluster_case1(data, labels, sigma, p, initial_type='rand')
    additional_function.scatter_plt(data, pred1, accuracy1,
                                    'MLE Estimated Gaussian Distributions with one center \n- historram and scatter plot- kmeans initialization')
    pred2, accuracy2 = additional_function.gauss_mle_cluster_case1(data, labels, sigma, p, initial_type='kmeans')
    additional_function.scatter_plt(data, pred2, accuracy2,
                                    'MLE Estimated Gaussian Distributions with one center \n- historram and scatter plot-  kmeans initialization')
    pred3, accuracy3 = additional_function.agg_clustering_case1(data,"euclidean")
    additional_function.scatter_plt(data, pred3, accuracy3,
                                    'agg Estimated Gaussian Distributions with one center \n- historram and scatter plot-  distance type min')
 #   pred4, accuracy4 = additional_function.agg_clustering_case1(data, labels, 2, 'max')
  #  additional_function.scatter_plt(data, pred4, accuracy4,
#                                   'agg Estimated Gaussian Distributions with one center \n- historram and scatter plot-  distance type max')
    pred5, accuracy5 = additional_function.agg_optimal_clustering_case1(data, labels, 2)
    additional_function.scatter_plt(data, pred5, accuracy5,
                                    'optimal agg Estimated Gaussian Distributions with one center \n- historram and scatter plot- distance type max')




def data_2D_multiple_centers():
    data, labels = make_blobs(n_samples=1000, centers=8, cluster_std=0.5, random_state=0, n_features=2)
    pred1, accuracy1=additional_function.gauss_mle_cluster_case1(data, labels, sigma, p, initial_type='rand')
    additional_function.scatter_plt(data, pred1, accuracy1,'MLE Estimated Gaussian Distributions with multiple centers \n- historram and scatter plot- kmeans initialization')
    pred2, accuracy2 = additional_function.gauss_mle_cluster_case1(data, labels, sigma, p, initial_type='kmeans')
    additional_function.scatter_plt(data, pred2, accuracy2, 'MLE Estimated Gaussian Distributions with multiple centers \n- historram and scatter plot-  kmeans initialization')
    pred3,accuracy3=additional_function.agg_clustering_case1(data,labels ,2, 'min')
    additional_function.scatter_plt(data, pred3, accuracy3, 'agg Estimated Gaussian Distributions with multiple centers \n- historram and scatter plot-  distance type min')
    pred4, accuracy4 = additional_function.agg_clustering_case1(data,labels, 2, 'max')
    additional_function.scatter_plt(data, pred4, accuracy4, 'agg Estimated Gaussian Distributions with multiple centers \n- historram and scatter plot-  distance type max')
    pred5, accuracy5 = additional_function.agg_optimal_clustering_case1(data, labels, 2)
    additional_function.scatter_plt(data, pred5, accuracy5,'optimal agg Estimated Gaussian Distributions with multiple centers \n- historram and scatter plot- distance type max')


def data_2D_circles():
    data, labels = make_circles(n_samples=1000, factor=0.5, noise=0.05, random_state=0)
    pred1, accuracy1 = additional_function.gauss_mle_cluster_case1(data, labels, sigma, p, initial_type='rand')
    additional_function.scatter_plt(data, pred1, accuracy1,
                                    'MLE Estimated Gaussian Distributions shaped like circels \n- historram and scatter plot- kmeans initialization')
    pred2, accuracy2 = additional_function.gauss_mle_cluster_case1(data, labels, sigma, p, initial_type='kmeans')
    additional_function.scatter_plt(data, pred2, accuracy2,
                                    'MLE Estimated Gaussian Distributions shaped like circels \n- historram and scatter plot-  kmeans initialization')
    pred3, accuracy3 = additional_function.agg_clustering_case1(data, labels, 2, 'min')
    additional_function.scatter_plt(data, pred3, accuracy3,
                                    'agg Estimated Gaussian Distributions shaped like circels \n- historram and scatter plot-  distance type min')
    pred4, accuracy4 = additional_function.agg_clustering_case1(data, labels, 2, 'max')
    additional_function.scatter_plt(data, pred4, accuracy4,
                                    'agg Estimated Gaussian Distributions shaped like circels \n- historram and scatter plot-  distance type max')
    pred5, accuracy5 = additional_function.agg_optimal_clustering_case1(data, labels, 2)
    additional_function.scatter_plt(data, pred5, accuracy5,
                                    'optimal agg Estimated Gaussian Distributions shaped like circels \n- historram and scatter plot- distance type max')

def data_2D_moons():
    data, labels = make_moons(n_samples=1000, noise=0.05)
    pred1, accuracy1 = additional_function.gauss_mle_cluster_case1(data, labels, sigma, p, initial_type='rand')
    additional_function.scatter_plt(data, pred1, accuracy1,
                                    'MLE Estimated Gaussian Distributions shaped like moones \n- historram and scatter plot- kmeans initialization')
    pred2, accuracy2 = additional_function.gauss_mle_cluster_case1(data, labels, sigma, p, initial_type='kmeans')
    additional_function.scatter_plt(data, pred2, accuracy2,
                                    'MLE Estimated Gaussian Distributions shaped like moones \n- historram and scatter plot-  kmeans initialization')
    pred3, accuracy3 = additional_function.agg_clustering_case1(data, labels, 2, 'min')
    additional_function.scatter_plt(data, pred3, accuracy3,
                                    'agg Estimated Gaussian Distributions shaped like moones \n- historram and scatter plot-  distance type min')
    pred4, accuracy4 = additional_function.agg_clustering_case1(data, labels, 2, 'max')
    additional_function.scatter_plt(data, pred4, accuracy4,
                                    'agg Estimated Gaussian Distributions shaped like moones \n- historram and scatter plot-  distance type max')
    pred5, accuracy5 = additional_function.agg_optimal_clustering_case1(data, labels, 2)
    additional_function.scatter_plt(data, pred5, accuracy5,
                                    'optimal agg Estimated Gaussian Distributions shaped like moones \n- historram and scatter plot- distance type max')


def unifornly_spred_square_data():
    square = np.random.rand(1000, 2)
    pred1 = additional_function.gauss_mle_cluster_case2(square, sigma, p, initial_type='rand')
    additional_function.scatter_plt(square, pred1, -1,
                                    'MLE Estimated uniformly spread square \n- historram and scatter plot- kmeans initialization')
    pred2 = additional_function.gauss_mle_cluster_case2(square, sigma, p, initial_type='kmeans')
    additional_function.scatter_plt(square, pred2, -1,
                                    'MLE Estimated uniformly spread square \n- historram and scatter plot-  kmeans initialization')
    pred3 = additional_function.agg_clustering_case2(square, 2, 'min')
    additional_function.scatter_plt(square, pred3,-1,
                                    'agg Estimated uniformly spread square \n- historram and scatter plot-  distance type min')
    pred4 = additional_function.agg_clustering_case2(square, 2, 'max')
    additional_function.scatter_plt(square, pred4, -1,
                                    'agg Estimated uniformly spread square \n- historram and scatter plot-  distance type max')
    pred5 = additional_function.agg_optimal_clustering_case2(square, 2)
    additional_function.scatter_plt(square, pred5, -1,
                                    'optimal agg Estimated uniformly spread square \n- historram and scatter plot- distance type max')


def unifornly_spred_parallelogram_data():
    parallelogram = np.random.rand(1000, 2)
    parallelogram[:, 0] = parallelogram[:, 0] + parallelogram[:, 1]
    pred1 = additional_function.gauss_mle_cluster_case2( parallelogram, sigma, p, initial_type='rand')
    additional_function.scatter_plt( parallelogram, pred1, -1,
                                    'MLE Estimated uniformly spread parallelogram \n- historram and scatter plot- kmeans initialization')
    pred2 = additional_function.gauss_mle_cluster_case2( parallelogram, sigma, p, initial_type='kmeans')
    additional_function.scatter_plt( parallelogram, pred2, -1,
                                    'MLE Estimated uniformly spread parallelogram \n- historram and scatter plot-  kmeans initialization')
    pred3 = additional_function.agg_clustering_case2( parallelogram, 2, 'min')
    additional_function.scatter_plt( parallelogram, pred3, -1,
                                    'agg Estimated uniformly spread parallelogram \n- historram and scatter plot-  distance type min')
    pred4 = additional_function.agg_clustering_case2( parallelogram, 2, 'max')
    additional_function.scatter_plt( parallelogram, pred4, -1,
                                    'agg Estimated uniformly spread parallelogram \n- historram and scatter plot-  distance type max')
    pred5 = additional_function.agg_optimal_clustering_case2( parallelogram, 2)
    additional_function.scatter_plt( parallelogram, pred5, -1,
                                    'optimal agg Estimated uniformly spread parallelogram \n- historram and scatter plot- distance type max')

def unifornly_spred_arrow_data():
    arrow = np.random.rand(1000, 2)
    arrow[:, 0] = np.cos(2 * np.pi * arrow[:, 1])
    pred1 = additional_function.gauss_mle_cluster_case2(arrow, sigma, p, initial_type='rand')
    additional_function.scatter_plt(arrow, pred1, -1,
                                    'MLE Estimated uniformly spread arrow \n- historram and scatter plot- kmeans initialization')
    pred2 = additional_function.gauss_mle_cluster_case2(arrow, sigma, p, initial_type='kmeans')
    additional_function.scatter_plt(arrow, pred2, -1,
                                    'MLE Estimated uniformly spread arrow \n- historram and scatter plot-  kmeans initialization')
    pred3 = additional_function.agg_clustering_case2(arrow, 2, 'min')
    additional_function.scatter_plt(arrow, pred3, -1,
                                    'agg Estimated uniformly spread arrow \n- historram and scatter plot-  distance type min')
    pred4 = additional_function.agg_clustering_case2(arrow, 2, 'max')
    additional_function.scatter_plt(arrow, pred4, -1,
                                    'agg Estimated uniformly spread arrow \n- historram and scatter plot-  distance type max')
    pred5 = additional_function.agg_optimal_clustering_case2(arrow, 2)
    additional_function.scatter_plt(arrow, pred5, -1,
                                    'optimal agg Estimated uniformly spread arrow \n- historram and scatter plot- distance type max')


def unifornly_spred_triangle_data():
    v1 = np.array([0, 0])
    v2 = np.array([0.2, 0.4])
    v3 = np.array([0.4, 0])
    u = np.random.uniform(size=(200, 1))
    v = np.random.uniform(size=(200, 1))
    triangle = (1 - np.sqrt(u)) * v1 + (np.sqrt(u) * (1 - v)) * v2 + (np.sqrt(u) * v) * v3
    pred1 = additional_function.gauss_mle_cluster_case2(triangle, sigma, p, initial_type='rand')
    additional_function.scatter_plt(triangle, pred1, -1,
                                    'MLE Estimated uniformly spread triangle \n- historram and scatter plot- kmeans initialization')
    pred2 = additional_function.gauss_mle_cluster_case2(triangle, sigma, p, initial_type='kmeans')
    additional_function.scatter_plt(triangle, pred2, -1,
                                    'MLE Estimated uniformly spread triangle \n- historram and scatter plot-  kmeans initialization')
    pred3 = additional_function.agg_clustering_case2(triangle, 2, 'min')
    additional_function.scatter_plt(triangle, pred3, -1,
                                    'agg Estimated uniformly spread triangle \n- historram and scatter plot-  distance type min')
    pred4 = additional_function.agg_clustering_case2(triangle, 2, 'max')
    additional_function.scatter_plt(triangle, pred4, -1,
                                    'agg Estimated uniformly spread triangle \n- historram and scatter plot-  distance type max')
    pred5 = additional_function.agg_optimal_clustering_case2(triangle, 2)
    additional_function.scatter_plt(triangle, pred5, -1,
                                    'optimal agg Estimated uniformly spread triangle \n- historram and scatter plot- distance type max')

gaussian_data_2D_one_center()

data_2D_multiple_centers()

data_2D_circles()

data_2D_moons()

unifornly_spred_square_data()

unifornly_spred_parallelogram_data()
unifornly_spred_arrow_data()

unifornly_spred_triangle_data()