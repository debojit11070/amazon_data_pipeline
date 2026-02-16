"""
A/B Testing Framework for Amazon Product Analytics
Statistical hypothesis testing for product strategies
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ABTestFramework:
    """
    Simple A/B testing framework for product analytics
    Performs statistical tests to compare groups
    """
    
    def __init__(self, alpha=0.05):
        """
        Initialize A/B test framework
        
        Args:
            alpha: Significance level (default 0.05 for 95% confidence)
        """
        self.alpha = alpha
        self.results = {}
    
    def compare_means(self, control, treatment, metric_name="Metric"):
        """
        Compare two groups using t-test
        
        Args:
            control: Array of control group values
            treatment: Array of treatment group values
            metric_name: Name of the metric being tested
        
        Returns:
            Dictionary with test results
        """
        # Calculate statistics
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        control_std = np.std(control)
        treatment_std = np.std(treatment)
        
        # Calculate lift
        lift = (treatment_mean - control_mean) / control_mean if control_mean != 0 else 0
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(treatment, control)
        
        # Determine significance
        is_significant = p_value < self.alpha
        
        # Store results
        results = {
            'metric': metric_name,
            'control_n': len(control),
            'treatment_n': len(treatment),
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'control_std': control_std,
            'treatment_std': treatment_std,
            'lift_pct': lift * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'confidence_level': (1 - self.alpha) * 100,
            'winner': 'Treatment' if is_significant and lift > 0 else 
                     'Control' if is_significant and lift < 0 else 
                     'No significant difference'
        }
        
        self.results[metric_name] = results
        
        # Print summary
        self._print_test_summary(results)
        
        return results
    
    def _print_test_summary(self, results):
        """Print formatted test results"""
        print("\n" + "="*60)
        print(f"📊 A/B Test Results: {results['metric']}")
        print("="*60)
        print(f"Control Group:")
        print(f"  • Sample Size: {results['control_n']}")
        print(f"  • Mean: {results['control_mean']:.3f}")
        print(f"  • Std Dev: {results['control_std']:.3f}")
        print(f"\nTreatment Group:")
        print(f"  • Sample Size: {results['treatment_n']}")
        print(f"  • Mean: {results['treatment_mean']:.3f}")
        print(f"  • Std Dev: {results['treatment_std']:.3f}")
        print(f"\nTest Statistics:")
        print(f"  • Lift: {results['lift_pct']:+.2f}%")
        print(f"  • P-value: {results['p_value']:.4f}")
        print(f"  • Significant: {'✅ YES' if results['is_significant'] else '❌ NO'}")
        print(f"  • Confidence: {results['confidence_level']:.0f}%")
        print(f"  • Winner: {results['winner']}")
        print("="*60)
    
    def visualize_comparison(self, control, treatment, metric_name="Metric"):
        """Create visualization comparing control vs treatment"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot comparison
        data_to_plot = [control, treatment]
        axes[0].boxplot(data_to_plot, labels=['Control', 'Treatment'])
        axes[0].set_title(f'{metric_name} Distribution Comparison')
        axes[0].set_ylabel(metric_name)
        axes[0].grid(True, alpha=0.3)
        
        # Bar chart of means with error bars
        means = [np.mean(control), np.mean(treatment)]
        stds = [np.std(control), np.std(treatment)]
        x_pos = [0, 1]
        
        axes[1].bar(x_pos, means, yerr=stds, capsize=10, 
                   color=['skyblue', 'lightcoral'], alpha=0.7)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(['Control', 'Treatment'])
        axes[1].set_title(f'Mean {metric_name} Comparison')
        axes[1].set_ylabel(metric_name)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save
        os.makedirs('outputs/ab_tests', exist_ok=True)
        filename = f"outputs/ab_tests/{metric_name.lower().replace(' ', '_')}_comparison.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization: {filename}")
        plt.close()


def run_ab_tests(input_file):
    """
    Run predefined A/B tests on the dataset
    
    Args:
        input_file: Path to processed data CSV
    """
    print("\n🧪 Starting A/B Testing Suite\n")
    
    # Load data
    df = pd.read_csv(input_file)
    
    # Initialize test framework
    tester = ABTestFramework(alpha=0.05)
    
    # --- TEST 1: High Discount vs Low Discount ---
    print("\n### TEST 1: Discount Strategy Impact ###")
    high_discount = df[df['discount_percentage'] > 50]
    low_discount = df[df['discount_percentage'] <= 50]
    
    test1 = tester.compare_means(
        low_discount['rating'].values,
        high_discount['rating'].values,
        metric_name="Product Rating (Discount Impact)"
    )
    tester.visualize_comparison(
        low_discount['rating'].values,
        high_discount['rating'].values,
        metric_name="Rating by Discount Level"
    )
    
    # --- TEST 2: Electronics vs Home Products ---
    print("\n### TEST 2: Category Performance Comparison ###")
    electronics = df[df['cat_l1'] == 'Electronics']
    home = df[df['cat_l1'] == 'Home&Kitchen']
    
    if len(electronics) > 0 and len(home) > 0:
        test2 = tester.compare_means(
            home['rating'].values,
            electronics['rating'].values,
            metric_name="Rating (Electronics vs Home)"
        )
        tester.visualize_comparison(
            home['rating'].values,
            electronics['rating'].values,
            metric_name="Electronics vs Home Rating"
        )
    
    # --- TEST 3: Budget vs Premium Products ---
    print("\n### TEST 3: Price Segment Impact ###")
    budget = df[df['discounted_price'] < 500]
    premium = df[df['discounted_price'] > 3000]
    
    if len(budget) > 0 and len(premium) > 0:
        test3 = tester.compare_means(
            budget['rating'].values,
            premium['rating'].values,
            metric_name="Rating (Budget vs Premium)"
        )
        tester.visualize_comparison(
            budget['rating'].values,
            premium['rating'].values,
            metric_name="Budget vs Premium Rating"
        )
    
    # --- TEST 4: High Popularity vs Low Popularity ---
    print("\n### TEST 4: Popularity Impact on Quality ###")
    high_pop = df[df['rating_count'] >= df['rating_count'].quantile(0.75)]
    low_pop = df[df['rating_count'] < df['rating_count'].quantile(0.25)]
    
    test4 = tester.compare_means(
        low_pop['rating'].values,
        high_pop['rating'].values,
        metric_name="Rating (Popularity Impact)"
    )
    tester.visualize_comparison(
        low_pop['rating'].values,
        high_pop['rating'].values,
        metric_name="Low vs High Popularity Rating"
    )
    
    # --- SUMMARY REPORT ---
    print("\n" + "="*60)
    print("📈 A/B TESTING SUMMARY REPORT")
    print("="*60)
    
    summary_data = []
    for test_name, result in tester.results.items():
        summary_data.append({
            'Test': test_name,
            'Control Mean': f"{result['control_mean']:.3f}",
            'Treatment Mean': f"{result['treatment_mean']:.3f}",
            'Lift %': f"{result['lift_pct']:+.2f}%",
            'P-value': f"{result['p_value']:.4f}",
            'Significant': '✅' if result['is_significant'] else '❌',
            'Winner': result['winner']
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save summary
    os.makedirs('outputs/ab_tests', exist_ok=True)
    summary_df.to_csv('outputs/ab_tests/ab_test_summary.csv', index=False)
    print("\n💾 Saved summary: outputs/ab_tests/ab_test_summary.csv")
    
    print("\n✅ A/B Testing Complete!")


if __name__ == "__main__":
    CLEANED_PATH = 'data/processed/cleaned_amazon.csv'
    
    if os.path.exists(CLEANED_PATH):
        run_ab_tests(CLEANED_PATH)
    else:
        print("❌ Error: Run governance.py first!")
        print(f"   Expected file: {CLEANED_PATH}")
