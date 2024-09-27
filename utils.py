import numpy as np
import pandas as pd
from PIL import Image
import urllib.request
import pyheif
from ultralytics import YOLO
import torchvision.transforms as transforms

def get_image(link):
    link_parts = link.split("/")
    file_id = link_parts[-2]
    parsed_link = f"https://drive.google.com/uc?id={file_id}"

    try:
        response = urllib.request.urlopen(parsed_link)
        heif_file = pyheif.read(response)

        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        image = image.convert('RGB')
        model = YOLO("yolov8n.pt")
        results = model(image, classes=0)  # or specify custom classes
        boxes = results[0].boxes
        coords = boxes.xyxy.tolist()[0]
        image = image.crop(coords)
        image = image.resize((224, 224))
        image_array = np.array(image)
        return image_array

    except Exception as e:
        print(f"Error: {e}, link: {link}")

def apply_random_augmentation(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToPILImage()
    ])
    augmented_image = transform(image)
    return np.array(augmented_image)

def augment_data(df, num_augmentations):
    original_length = len(df)
    if original_length == 0:
        return df
    augmented_rows = []
    
    for i in range(num_augmentations):
        row_index = i % original_length
        row = df.iloc[row_index].copy()
        row['Front Image'] = apply_random_augmentation(Image.fromarray(row['Front Image']))
        row['Back Image'] = apply_random_augmentation(Image.fromarray(row['Back Image']))
        augmented_rows.append(row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    return augmented_df

def process_df(df):
    for index, row in df.iterrows():
        print(index)
        front_image = get_image(row['Front Image'])
        back_image = get_image(row['Back Image'])
        if front_image is None or back_image is None:
            df.drop(index, inplace=True)
            continue
        df.at[index, 'Front Image'] = front_image
        df.at[index, 'Back Image'] = back_image
        df.at[index, 'Training Body Fat %'] = float(row['Training Body Fat %'][:-1])
    return df

def distribute_body_fat(df):
    df = df.copy()
    df['Body Fat Bucket'] = pd.cut(df['Training Body Fat %'], bins=range(4, 30, 2), labels=False)
    bucket_counts = df['Body Fat Bucket'].value_counts().sort_values()
    target_count = bucket_counts.iloc[-1]

    for bucket, count in bucket_counts.items():
        if count < target_count:
            bucket_df = df[df['Body Fat Bucket'] == bucket].copy()
            num_augmentations = target_count - count
            augmented_df = augment_data(bucket_df, num_augmentations)
            df = pd.concat([df, augmented_df], ignore_index=True)

    df.drop('Body Fat Bucket', axis=1, inplace=True)
    return df

def process_data(df):
    X_front_images = []
    X_back_images = []
    X_tabular = []
    Y_body_fat = []

    for index, row in df.iterrows():
        X_front_images.append((row['Front Image'].astype(np.float32) / 255) - 0.5)
        X_back_images.append((row['Back Image'].astype(np.float32) / 255) - 0.5)
        X_tabular.append([float(row['Height']), float(row['Weight']), float(row['Waist'])/float(row['Hips'])])
        Y_body_fat.append(float(row['Training Body Fat %']))
    
    X_front_images = np.array(X_front_images)
    X_back_images = np.array(X_back_images)
    X_tabular = np.array(X_tabular)
    Y_body_fat = np.array(Y_body_fat)

    return X_front_images, X_back_images, X_tabular, Y_body_fat