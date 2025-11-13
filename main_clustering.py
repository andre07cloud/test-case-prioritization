
from src.common.flattened_data import flattened_features
from src.common.hierarchical_clustering import *


if __name__ == "__main__":
    config_file = "config.yaml"
    #df = flattened_features(config_file)
    # with open("flattened_scene_features.csv", "w") as f:
    #     df.to_csv(f, index=False)
    print("Flattened scene features saved to flattened_scene_features.csv")
    data1 = "flattened_scene_features.csv"

    #hierarchical_clustering(data1)

    data2 = "features_avec_clusters_hc.csv"

    diversity_clustering(data2)