import pandas as pd
import numpy as np
import datetime as dt
from typing import Sequence
import os
import sys
import json
import logging
import click
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from bac.util.config import parse_config


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# TODO: Refactor main Kmeans portion as class and put in models folder
# TODO: Fix the doc strings and comment better


def get_centroids(km):
    """km: K-Means model object
    """
    # The cluster each point or observation belongs to
    cluster_labels = km.labels_
    # The means of the points in each cluster (centroids)
    centroids = km.cluster_centers_
    return cluster_labels, centroids


def make_centroid_geodata(centroids) -> Sequence:
    """Format centroids to graph as geodata

    Args:
        centroids ([type]): [description]

    Returns:
        [type]: a list that can be cut and pasted into a geojson
    """
    features = []
    for lon, lat in centroids:
        geo = {}
        geo["type"] = "Feature"
        geo["geometry"] = {"type":"Point", "coordinates": [lon, lat]}
        features.append(geo)
    return features


def make_geojson(features) -> dict:
    """Complete geojson"""
    geo={}
    geo["type"]="FeatureCollection"
    geo["features"]=features
    return geo


def find_optimal_k(k, X, mod, step, km_params):
    # Find Optimal K
    sum_squared_dist = []
    for kidx in range(0, k, step):
        if kidx%mod==0:
            logging.info(f'{dt.datetime.now()}: KMeans iteration {kidx}...')
        km_params['n_clusters'] = kidx+1
        km = KMeans(**km_params).fit(X)
        sum_squared_dist.append(km.inertia_)
    return sum_squared_dist


def plot_kmeans_ssd(k, sum_squared_dist, step):
    # TODO: Save figures
    #plt.plot(range(len(sum_squared_dist)), np.log(sum_squared_dist), 'bx-')
    plt.plot(np.arange(0, k, step)+1, np.log(sum_squared_dist), 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def save_geojson(k: int, geojson: dict, output_fpath: str) -> None:
    """Save geojson data"""
    filename = f'geojson_centroids_k{k}_{dt.date.today()}.json'
    output_filepath = os.path.join(output_fpath, filename)
    with open(output_filepath, 'w') as f:
        json.dump(geojson, f)
    logging.info('\nSaved geojson')


@click.command()
@click.argument("config", type=click.Path(exists=True), default="/mnt/configs/kmeans.yml")
def main(config):
    """Given lat-lon data for each BrAC observation,
    assign observations to clusters by the Kmeans algorithm,
    then format into a geojson, in order to visualize using MapBox API

    Args:
        config (dict): includes io filepaths & kmeans-kwargs
    """
    
    config = parse_config(config)
    input_fpath = config["input_fpath"]
    output_fpath = config["output_fpath"]
    km_params = config["km_params"]
    
    
    df = pd.read_csv(input_fpath, sep=',', header=0)
    logging.info(df.head(3))
    
    X = df[['longitude','latitude']]
    
    k = 120# Number of clusters
    step = 1# step--> increments k
    mod = 2# modulo--> determines verbosity/logging frequency
    
    sum_squared_dist = find_optimal_k(k, X, mod, step, km_params)
    plot_kmeans_ssd(k, sum_squared_dist, step)
    
    km_params['n_clusters']=k
    km = KMeans(**km_params).fit(X)
    logging.info(f'Fit model with {k} centroids')
    
    cluster_labels, centroids = get_centroids(km)
    features = make_centroid_geodata(centroids)
    geojson = make_geojson(features)

    save_geojson(k, geojson, output_fpath)
    
    
if __name__ == "__main__":
    main()


