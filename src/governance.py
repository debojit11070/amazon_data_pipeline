import pandas as pd
import numpy as np
import os

def run_governance_pipeline(input_file, output_file):
    print(f"🚀 Starting Governance Pipeline on: {input_file}")
    
    # Load raw data
    df = pd.read_csv(input_file)
    
    # --- 1. Price Governance ---
    # Convert ₹ strings to floats (e.g., '₹1,099' -> 1099.0)
    for col in ['discounted_price', 'actual_price']:
        df[col] = df[col].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
    
    # --- 2. Discount Integrity ---
    # Convert '64%' to 64.0
    df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '').astype(float)
    
    # RECALCULATE: Hard logic check
    # Check if stated discount matches (actual - discounted) / actual
    df['expected_discount'] = round(100 - (df['discounted_price'] / df['actual_price'] * 100), 0)
    
    # Flag inconsistencies > 3 percentage points
    df['discount_inconsistency_flag'] = (abs(df['discount_percentage'] - df['expected_discount']) > 3).astype(int)
    
    # --- 3. Rating & Engagement Governance ---
    # # Handle the 'rating' field (handle potential non-numeric junk)
    # df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    # df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').fillna(0).astype(int)
    
    # --- 3. Rating & Engagement Governance ---
    # Handle the 'rating' field (handle potential non-numeric junk)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # NEW FIX: Convert to string, replace commas, then replace 'nan' strings with '0'
    df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '')
    df['rating_count'] = df['rating_count'].replace('nan', '0').replace('', '0')
    
    # Now it is safe to convert to integer
    df['rating_count'] = df['rating_count'].astype(int)
    
    # --- 4. Category Taxonomy Expansion ---
    # Split "Electronics|Accessories|Cables" into L1, L2, L3
    category_split = df['category'].str.split('|', expand=True)
    df['cat_l1'] = category_split[0]
    df['cat_l2'] = category_split[1]
    df['cat_l3'] = category_split[2]
    
    # --- 5. Satisfaction Risk Labeling ---
    # Define "At-Risk" as rating < 4.0 with at least 50 votes
    df['at_risk'] = ((df['rating'] < 4.0) & (df['rating_count'] >= 50)).astype(int)
    
    # Save to Processed folder
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"✅ Governance Complete!")
    print(f"📊 Summary: {len(df)} products governed.")
    print(f"⚠️ Flagged {df['discount_inconsistency_flag'].sum()} products for discount mismatch.")

if __name__ == "__main__":
    # Ensure paths match your structure
    RAW_PATH = 'data/raw/amazon.csv'
    PROCESSED_PATH = 'data/processed/cleaned_amazon.csv'
    
    if os.path.exists(RAW_PATH):
        run_governance_pipeline(RAW_PATH, PROCESSED_PATH)
    else:
        print(f"❌ Error: {RAW_PATH} not found. Did you copy it into the folder?")