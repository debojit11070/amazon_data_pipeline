import subprocess
import sys
import os

def run_step(script_path, description):
    print(f"\n{'='*50}")
    print(f"🚀 STEP: {description}")
    print(f"{'='*50}")
    
    # Run the python script as a subprocess
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ ERROR: {description} failed. Stopping pipeline.")
        sys.exit(1)
    else:
        print(f"✅ SUCCESS: {description} completed.")

def main():
    # Define the order of operations
    steps = [
        ("src/governance.py", "Data Governance & Cleaning"),
        ("src/eda.py", "Exploratory Data Analysis"),
        ("src/ab_testing.py", "A/B Hypothesis Testing"),
        ("src/sentiment_analysis.py", "Sentiment Analysis (NLP)"),
        ("src/feature_engineering.py", "Feature Engineering"),
        ("src/train_model.py", "Model Training")
    ]

    print("🏁 Starting Amazon Data Engineering Pipeline...")

    for script, desc in steps:
        if os.path.exists(script):
            run_step(script, desc)
        else:
            print(f"⚠️ Warning: Could not find {script}")

    print("\n" + "✨"*20)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("You can now run 'streamlit run app/main.py' to see the dashboard.")
    print("✨"*20)

if __name__ == "__main__":
    main()