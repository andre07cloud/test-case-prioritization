import os
import random
import shutil
import yaml
import sys 



# ++++++++++++++++++++
# CONFIGURATION
# ++++++++++++++++++++
OLD_DATASET_DIR = "dataset_wood_old"  # Original dataset directory
NEW_DATASET_DIR = "dataset_wood_new"  # New dataset directory
MERGED_DATASET_DIR = "dataset_wood_all"  # Merged dataset directory
IMG_DIR1 = "data/images_uc1"
IMG_DIR2 = "data/images_uc2"
LABEL_DIR1 = "data/labels_uc1"
LABEL_DIR2 = "data/labels_uc2"
OUT_DIR = "data/dataset_wood"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
TEST_RATIO = 0.1
YAMEL_FILE = "wood.yaml"









# ++++++++++++++++++++
# MOVE FILES
# ++++++++++++++++++++

def move_files(files, img_dest, labe_dest, image_path=None, label_path=None):
    
    for f in files:
        img_src = os.path.join(image_path, f)
        print(f"img_src: {img_src}")
        label_src = os.path.join(label_path, os.path.splitext(f)[0] + '.txt')

        img_dst = os.path.join(img_dest, f)
        label_dst = os.path.join(labe_dest, os.path.splitext(f)[0] + '.txt')

        shutil.copy(img_src, img_dst)
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)

# ++++++++++++++++++++
# COPY DATASET
# ++++++++++++++++++++

def copy_files(file_list, split):
    for img_src, label_src in file_list:
        img_name = os.path.basename(img_src)
        label_name = os.path.basename(label_src)

        img_dist = os.path.join(MERGED_DATASET_DIR, "images", split, img_name)
        label_dist = os.path.join(MERGED_DATASET_DIR, "labels", split, label_name)

        shutil.copy(img_src, img_dist)
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dist)




def split_dataset(dataset_old=OLD_DATASET_DIR, use_case_path=None, img_path=None, label_path=None):

    # ++++++++++++++++++++
        # THE STRCUTURE
    # ++++++++++++++++++++
    print("Creating dataset structure...")
    print(f"Using old dataset from: {dataset_old}")
    print(f"Using new dataset from: {NEW_DATASET_DIR}")
    print(f"use case path: {use_case_path}")
    root_path = f"data/{use_case_path}/{dataset_old}"
    train_img_dir = os.path.join(root_path, "train/images")
    train_label_dir = os.path.join(root_path, "train/labels")
    val_img_dir = os.path.join(root_path, "val/images")
    val_label_dir = os.path.join(root_path, "val/labels")
    #test_img_dir = os.path.join(root_path, "test/images")
    #test_label_dir = os.path.join(root_path, "test/labels")

    for d in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        os.makedirs(d, exist_ok=True)
    
    # ++++++++++++++++++++
        # LIST IMAGES
    # ++++++++++++++++++++

    all_images = [f for f in os.listdir(img_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(all_images)

    # split train/val
    split_index = int(len(all_images) * TRAIN_RATIO)
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]

    # n_total = len(all_images)
    # n_train = int(n_total * TRAIN_RATIO)
    # print(f"n_total: {n_total}, n_train: {n_train}")
    # n_val = int(n_total * VAL_RATIO)
    # print(f"n_total: {n_total}, n_val: {n_val}")
    # n_test = n_total - n_train - n_val  # Ensure all images are accounted for
    # print(f"n_total: {n_total}, n_test: {n_test}")
    # train_images = all_images[:n_train]
    # print(f"train_images: {len(train_images)}")
    # print(train_images)
    # val_images = all_images[n_train:n_train + n_val]
    # print(f"val_images: {len(val_images)}")
    # print(val_images)
    # test_images = all_images[n_train + n_val:n_train + n_val + n_test]
    # print(f"test_images: {len(test_images)}")
    # print(test_images)

    move_files(train_images, train_img_dir, train_label_dir, image_path=img_path, label_path=label_path)
    move_files(val_images, val_img_dir, val_label_dir, image_path=img_path, label_path=label_path)
    #move_files(test_images, test_img_dir, test_label_dir, image_path=img_path, label_path=label_path)


    print(f"Dataset created at {dataset_old} to Taotal {len(all_images)} with {len(train_images)} training and {len(val_images)} validation images. ")






def split_dataset_merged(dataset_old=OLD_DATASET_DIR, dataset_new=NEW_DATASET_DIR):

    for split in ["train", "val"]:
        for subfolder in ["images", "labels"]:
            dir_path = os.path.join(MERGED_DATASET_DIR, subfolder, split)
            os.makedirs(dir_path, exist_ok=True)

    all_image_merged = []

    for dataset in [dataset_old, dataset_new]:
        for split in ["train", "val"]:
            img_dir = os.path.join(dataset, "images", split)
            label_dir = os.path.join(dataset, "labels", split)

            if os.path.exists(img_dir):
                for f in os.listdir(img_dir):
                    if f.lower().endswith((".png", ".jpg", ".jpeg" )):
                        img_path = os.path.join(img_dir, f)
                        label_path = os.path.join(label_dir, os.path.splitext(f)[0] + ".txt")
                        all_image_merged.append((img_path, label_path))

    print(f"Total images collected: {len(all_image_merged)}")

    # Shuffle the combined list of images
    random.shuffle(all_image_merged)

        # split train/val merged
    split_index = int(len(all_image_merged) * TRAIN_RATIO)
    train_images = all_image_merged[:split_index]
    val_images = all_image_merged[split_index:]

    copy_files(train_images, "train")
    copy_files(val_images, "val")

    print(f"Dataset created at {MERGED_DATASET_DIR} with {len(train_images)} training and {len(val_images)} validation images.")

