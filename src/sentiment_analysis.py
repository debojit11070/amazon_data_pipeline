import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

def run_sentiment_pipeline(input_file, output_file):
    print(f"🧠 Starting Sentiment Analysis on: {input_file}")
    
    # Load the CLEANED data from your governance step
    df = pd.read_csv(input_file)
    
    # Initialize VADER
    analyzer = SentimentIntensityAnalyzer()
    
    # Function to calculate the sentiment score
    def get_sentiment(text):
        if pd.isna(text):
            return 0
        # The 'compound' score is the overall sentiment from -1 (negative) to 1 (positive)
        return analyzer.polarity_scores(str(text))['compound']

    print("⏳ Analyzing 1,465 reviews... (this will be fast)")
    df['sentiment_score'] = df['review_content'].apply(get_sentiment)
    
    # Create labels based on the score
    df['sentiment_label'] = df['sentiment_score'].apply(
        lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
    )
    
    # Save to a new 'features' folder (this is now "Feature Engineering")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"✅ Analysis Complete!")
    print(f"📈 Average Marketplace Sentiment: {df['sentiment_score'].mean():.2f}")
    print(f"📊 Positive: {len(df[df['sentiment_label'] == 'Positive'])} | Negative: {len(df[df['sentiment_label'] == 'Negative'])}")

if __name__ == "__main__":
    CLEANED_PATH = 'data/processed/cleaned_amazon.csv'
    FEATURE_PATH = 'data/features/amazon_with_sentiment.csv'
    
    if os.path.exists(CLEANED_PATH):
        run_sentiment_pipeline(CLEANED_PATH, FEATURE_PATH)
    else:
        print("❌ Error: Run governance.py first!")