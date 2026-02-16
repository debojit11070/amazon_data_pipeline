import pandas as pd
import numpy as np
import os

def run_feature_engineering(input_file, output_file):
    df = pd.read_csv(input_file)

    # --- 1. Price Features ---
    df['discount_ratio'] = df['discounted_price'] / df['actual_price']
    df['discount_percentage_clean'] = (1 - df['discount_ratio']) * 100
    
    # --- 2. Engagement Features ---
    df['popularity_log'] = np.log1p(df['rating_count'])

    # --- 3. The Target (The 'Label' we want to predict) ---
    # We keep this logic, but we will HIDE the components during training
    df['normalized_rating'] = (df['rating'] - 1) / 4 
    df['sentiment_gap'] = df['normalized_rating'] - ((df['sentiment_score'] + 1) / 2)
    df['is_high_risk'] = ((df['rating'] < 3.8) | (df['sentiment_gap'] > 0.3)).astype(int)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✅ Features Engineered. Risk ratio: {df['is_high_risk'].mean():.2%}")

if __name__ == "__main__":
    run_feature_engineering('data/features/amazon_with_sentiment.csv', 'data/features/final_training_data.csv')