import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_risk_model(input_file):
    df = pd.read_csv(input_file)

    # 🚨 FIX: We REMOVE 'rating' and 'sentiment_score' from this list
    # This forces the model to learn if 'Price' or 'Popularity' predicts Risk
    features = ['actual_price', 'discounted_price', 'discount_ratio', 'popularity_log']
    
    X = df[features]
    y = df['is_high_risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"🚀 Realistic Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(classification_report(y_test, predictions))

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/satisfaction_model.pkl')
    joblib.dump(features, 'models/feature_list.pkl')

if __name__ == "__main__":
    train_risk_model('data/features/final_training_data.csv')