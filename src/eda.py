import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(input_file):
    print(f"📊 Running EDA on: {input_file}")
    df = pd.read_csv(input_file)
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # --- 1. Basic Stats & Table Export ---
    print("\n--- Generating Data Overview Table ---")
    stats_table = df[['actual_price', 'discounted_price', 'rating', 'rating_count']].describe()
    
    # Save the table to CSV
    stats_table.to_csv('outputs/data_summary_stats.csv')
    print("✅ Saved Table: outputs/data_summary_stats.csv")
    print(stats_table)

    # --- 2. Category Distribution (Top 5) ---
    plt.figure(figsize=(10, 6))
    df['cat_l1'].value_counts().head(5).plot(kind='bar', color='skyblue')
    plt.title('Top 5 Product Categories')
    plt.ylabel('Number of Products')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/category_dist.png')
    print("✅ Saved Chart: outputs/category_dist.png")

    # --- 3. Price vs Rating Correlation ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='actual_price', y='rating', alpha=0.5)
    plt.xscale('log') 
    plt.title('Price vs. Customer Rating (Log Scale)')
    plt.savefig('outputs/price_vs_rating.png')
    print("✅ Saved Chart: outputs/price_vs_rating.png")

    # --- 4. Governance Check Summary ---
    inconsistent = df['discount_inconsistency_flag'].sum()
    with open('outputs/governance_summary.txt', 'w') as f:
        f.write(f"Total Products Analyzed: {len(df)}\n")
        f.write(f"Inconsistent Discounts Found: {inconsistent}\n")
        f.write(f"Average Rating: {df['rating'].mean():.2f}\n")
    print("✅ Saved Summary: outputs/governance_summary.txt")

if __name__ == "__main__":
    CLEANED_PATH = 'data/processed/cleaned_amazon.csv'
    if os.path.exists(CLEANED_PATH):
        run_eda(CLEANED_PATH)
    else:
        print("❌ Error: Run governance.py first!")