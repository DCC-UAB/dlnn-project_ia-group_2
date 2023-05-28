import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_val_split(captions_file):
    
    # Split data into train val and test sets
    df_captions = pd.read_csv(captions_file)
    unique_images = df_captions['image'].unique()
    train_images, testval_images = train_test_split(unique_images, test_size=0.25, random_state=42)
    val_images, test_images = train_test_split(testval_images, test_size=0.5, random_state=42)
    train_df = df_captions[df_captions['image'].isin(train_images)]
    val_df = df_captions[df_captions['image'].isin(val_images)]
    test_df = df_captions[df_captions['image'].isin(test_images)]
    
    return train_df, val_df, test_df