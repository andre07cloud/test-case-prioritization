import pandas as pd
import json
import random
from .load_config_file import load_config_file
from .perception_difficulty import difficulty_evaluation


def flattened_features(config_file):
    """
    Flatten the scene features from the config file into a single dictionary.
    """
    scene_config = load_config_file(config_file)
    scene_features_file = scene_config['scene_features']
    print(f"Loading scene features from: {scene_features_file}")

    with open(scene_features_file, 'r') as f:
        scene_features = json.load(f)
    
    all_images_data = []

    for image_id, data in scene_features.items():
        row_data = {'image_id': image_id}
        poses = data['poses']
        if 'lighting' in data:
            row_data['lighting'] = data['lighting']

        for i, pose in enumerate(poses):

            obj_num = i + 1

            #flatten pose dictionary
            row_data[f'pos_x{obj_num}'] = pose['position'][0]
            row_data[f'pos_y{obj_num}'] = pose['position'][1]
            row_data[f'pos_z{obj_num}'] = pose['position'][2]
            #flatten rotation dictionary
            row_data[f'rot_x{obj_num}'] = pose['orientation'][0]
            row_data[f'rot_y{obj_num}'] = pose['orientation'][1]
            row_data[f'rot_z{obj_num}'] = pose['orientation'][2]
            row_data[f'rot_w{obj_num}'] = pose['orientation'][3]

        image_path = data['image']
        label_path = data['label']
        row_data['difficulty'] = difficulty_evaluation(image_path, label_path, config_file)
        all_images_data.append(row_data)

    df = pd.DataFrame(all_images_data)
    print(f"Dataframe created successfully with shape: {df.shape}")
    print(df.head())
    return df