from src.common.split_range_data import split_dataset, split_dataset_merged, OLD_DATASET_DIR, NEW_DATASET_DIR, IMG_DIR1, IMG_DIR2, LABEL_DIR1, LABEL_DIR2, YAMEL_FILE, MERGED_DATASET_DIR
import os
import sys
import yaml



if __name__ == "__main__":

    if sys.argv[1] == "split":
        print("Splitting dataset...")
        print("Use case:", sys.argv[2], type(sys.argv[2]))
        if sys.argv[2] == "uc1":

            if os.path.exists(OLD_DATASET_DIR):
                split_dataset(dataset_old=NEW_DATASET_DIR, use_case_path=sys.argv[2], img_path=IMG_DIR1, label_path=LABEL_DIR1)
            else:
                print(f"Directory {OLD_DATASET_DIR} does not exist. Please check the path.")
                split_dataset(use_case_path=sys.argv[2], img_path=IMG_DIR1, label_path=LABEL_DIR1)
        elif sys.argv[2] == "uc2":
            if os.path.exists(OLD_DATASET_DIR):
                split_dataset(dataset_old=OLD_DATASET_DIR, use_case_path=sys.argv[2], img_path=IMG_DIR2, label_path=LABEL_DIR2)
            else:
                print(f"Directory {OLD_DATASET_DIR} does not exist. Please check the path.")
                split_dataset(use_case_path=sys.argv[2], img_path=IMG_DIR2, label_path=LABEL_DIR2)

    elif sys.argv[1] == "merge":
        split_dataset_merged()


    # ++++++++++++++++++++
    # CREATE YAML
    # ++++++++++++++++++++
    if sys.argv[1] == "split":
        
        root_path = f"{sys.argv[2]}/{OLD_DATASET_DIR}"
        data = {
            #'path': root_path,
            'task': 'obb',  # obb for Oriented Bounding Box, aabb for Axis-Aligned Bounding Box
            'train': os.path.join(root_path, 'train/images'),
            'val': os.path.join(root_path, 'val/images'),
            #'test': os.path.join(root_path, 'test/images'),
            'nc': 1,  # or the actual number of classes
            'names': ['wood']  # adapt to your case
        }
    else:
        if sys.argv[1] == "merge":
            root_path = f"{sys.argv[2]}/{MERGED_DATASET_DIR}"
            data = {
                'task': 'obb',  # obb for Oriented Bounding Box, aabb for Axis-Aligned Bounding Box
                'train': os.path.join(root_path, 'train/images'),
                'val': os.path.join(root_path, 'val/images'),
                #'test': os.path.join(root_path, 'test/images'),
                'nc': 1,  # or the actual number of classes
                'names': ['wood']  # adapt to your case
            }

    with open(f"data/{sys.argv[2]}_{YAMEL_FILE}", 'w') as f:
        yaml.dump(data, f)

    print(f"YAML file created at data/{sys.argv[2]}_{YAMEL_FILE}")