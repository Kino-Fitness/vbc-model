import numpy as np
import pandas as pd
from PIL import Image
import urllib.request
from pillow_heif import register_heif_opener
from ultralytics import YOLO
import torchvision.transforms as transforms
import os

register_heif_opener()

def is_valid_numeric_row(row):
    """Check if all required numeric columns have valid values."""
    required_columns = [
        'Height', 'Weight', 'Training Body Fat %', 
        'Training Muscle Mass', 'Training Bone Mass', 
        'Training Bone Density', 'Waist', 'Hip (bone)'
    ]
    
    try:
        for col in required_columns:
            if col not in row:
                return False
            # Handle percentage values
            value = float(row[col].strip('%')) if isinstance(row[col], str) and '%' in row[col] else float(row[col])
            if np.isnan(value) or value == 0:  # Check for NaN and zero values
                return False
        return True
    except (ValueError, TypeError):
        return False

def get_image(link):
    if not isinstance(link, str):  # Basic link validation
        return None
        
    try:
        link_parts = link.split("/")
        file_id = link_parts[-2]
        parsed_link = f"https://drive.google.com/uc?id={file_id}"

        response = urllib.request.urlopen(parsed_link)
        image = Image.open(response)
        image = image.convert('RGB')
        
        model = YOLO("yolov8n.pt")
        results = model(image, classes=0)
        boxes = results[0].boxes
        coords = boxes.xyxy.tolist()[0]
        image = image.crop(coords)
        image = image.resize((224, 224))
        return image

    except Exception as e:
        print(f"Error: {e}, link: {link}")
        return None

def apply_random_augmentation(image):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
        front_pil = Image.fromarray(row['Front Image'].astype('uint8'))
        back_pil = Image.fromarray(row['Back Image'].astype('uint8'))
        row['Front Image'] = apply_random_augmentation(front_pil)
        row['Back Image'] = apply_random_augmentation(back_pil)
        augmented_rows.append(row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    return augmented_df

def process_df(df):
    for index, row in df.iterrows():
        print(index)
        # Check if numeric data is valid before processing
        if not is_valid_numeric_row(row):
            df.drop(index, inplace=True)
            continue
            
        front_image = get_image(row['Front Image'])
        back_image = get_image(row['Back Image'])
        if front_image is None or back_image is None:
            df.drop(index, inplace=True)
            continue
            
        df.at[index, 'Front Image'] = np.array(front_image)
        df.at[index, 'Back Image'] = np.array(back_image)
        df.at[index, 'Training Body Fat %'] = float(row['Training Body Fat %'].strip('%'))
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
    Y_muscle_mass = []
    Y_bone_mass = []
    Y_bone_density = []

    X_tabularBF =[]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for index, row in df.iterrows():
        # Skip row if any required data is invalid
        if not is_valid_numeric_row(row):
            continue
            
        # Process images
        front_image = np.array(row['Front Image']).astype(np.float32) / 255.0
        front_image = (front_image - mean) / std
        X_front_images.append(front_image)

        back_image = np.array(row['Back Image']).astype(np.float32) / 255.0
        back_image = (back_image - mean) / std
        X_back_images.append(back_image)

        X_tabularBF.append([
            float(row['Height']), 
            float(row['Weight'])
        ])

        # Process tabular data
        waist = float(row['Waist'])
        hips = float(row['Hip (bone)'])
        X_tabular.append([
            float(row['Height']), 
            float(row['Weight']), 
            waist / hips
        ])

        # Process labels
        Y_body_fat.append(float(row['Training Body Fat %']))
        Y_bone_mass.append(float(row['Training Bone Mass']))
        Y_muscle_mass.append(float(row['Training Muscle Mass']))
        Y_bone_density.append(float(row['Training Bone Density']))

    # Convert to numpy arrays
    X_front_images = np.array(X_front_images)
    X_back_images = np.array(X_back_images)
    X_tabular = np.array(X_tabular)
    Y_body_fat = np.array(Y_body_fat)
    Y_muscle_mass = np.array(Y_muscle_mass)
    Y_bone_mass = np.array(Y_bone_mass)
    Y_bone_density = np.array(Y_bone_density)

    return X_front_images, X_back_images,X_tabularBF, X_tabular, Y_body_fat, Y_muscle_mass, Y_bone_mass, Y_bone_density