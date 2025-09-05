"""
Customer Feedback Analysis Pipeline
Main script for analyzing customer feedback data with theme mining, trends, anomalies, and recommendations
"""

import pandas as pd
import numpy as np
import json
import os
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats
from tqdm.auto import tqdm
from concurrent import futures
import google.generativeai as genai
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Import prompts
from prompts import (
    THEME_EXTRACTION_PROMPT,
    THEME_CONSOLIDATION_PROMPT,
    TREND_SUMMARY_PROMPT,
    RECOMMENDATION_PROMPT
)

# ========================================
# Configuration
# ========================================

# Paths
DATA_DIRECTORY = "/Users/goyolozano/Desktop/CG Feedback/Data/"
OUTPUT_DIRECTORY = "/Users/goyolozano/Desktop/CG Feedback/Outputs/"
DOTENV_PATH = "/Users/goyolozano/Desktop/CG Feedback/.env"

# Create output directory if it doesn't exist
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

# LLM Configuration
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
MAX_WORKERS = 30

# ========================================
# Initialization
# ========================================

def initialize_llm():
    """Initialize Google Gemini LLM"""
    print("=" * 60)
    print("INITIALIZING LLM")
    print("=" * 60)
    
    load_dotenv(dotenv_path=DOTENV_PATH)
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    
    if GOOGLE_API_KEY is None:
        raise ValueError("ERROR: GOOGLE_API_KEY not found in .env file")
    
    print("✓ API key loaded successfully")
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    print(f"✓ Initialized {GEMINI_MODEL_NAME}")
    return model

def load_data():
    """Load feedback and product usage data"""
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    print(f"Loading data from: {DATA_DIRECTORY}")
    
    product_usage_df = pd.read_csv(DATA_DIRECTORY + "product_usage.csv")
    feedback_df = pd.read_json(DATA_DIRECTORY + "feedback.jsonl", lines=True)
    
    print(f"✓ Product usage data: {product_usage_df.shape[0]} rows, {product_usage_df.shape[1]} columns")
    print(f"✓ Feedback data: {feedback_df.shape[0]} rows, {feedback_df.shape[1]} columns")
    print(f"  - Unique customers in feedback: {feedback_df['customer_id'].nunique()}")
    print(f"  - Unique customers in product usage: {product_usage_df['customer_id'].nunique()}")
    print(f"  - Matching customer IDs: {len(set(feedback_df['customer_id']) & set(product_usage_df['customer_id']))}")
    
    return feedback_df, product_usage_df

# ========================================
# Theme Mining Functions
# ========================================

def analyze_feedback_with_cache(message, model, cache):
    """Analyze a single feedback message using LLM with caching"""
    if message in cache:
        return cache[message]
    
    prompt = THEME_EXTRACTION_PROMPT.format(message=message)
    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned_response)
        
        theme = result.get("theme", "Error: No Theme")
        sentiment = result.get("sentiment", "Error: No Sentiment")
        
        cache[message] = (theme, sentiment)
        return theme, sentiment
        
    except Exception as e:
        print(f"  ! Error processing message: '{message[:50]}...'. Error: {e}")
        return "Error: Parsing Failed", "Error: Parsing Failed"

def extract_themes(feedback_df, model):
    """Extract themes and sentiments from all feedback messages"""
    print("\n" + "=" * 60)
    print("THEME EXTRACTION")
    print("=" * 60)
    
    feedback_analysis_df = feedback_df.copy()
    unique_messages = feedback_analysis_df['message'].unique().tolist()
    
    print(f"Processing {len(unique_messages)} unique messages...")
    print(f"Using {MAX_WORKERS} parallel workers")
    
    analysis_cache = {}
    results_map = {}
    
    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_message = {
            executor.submit(analyze_feedback_with_cache, msg, model, analysis_cache): msg 
            for msg in unique_messages
        }
        
        with tqdm(total=len(unique_messages), desc="Extracting themes") as pbar:
            for future in futures.as_completed(future_to_message):
                message = future_to_message[future]
                try:
                    theme, sentiment = future.result()
                    results_map[message] = (theme, sentiment)
                except Exception as exc:
                    print(f"  ! Exception for message: {exc}")
                    results_map[message] = ("Error: Exception", "Error: Exception")
                pbar.update(1)
    
    # Map results back to DataFrame
    mapped_results = feedback_analysis_df['message'].map(results_map)
    feedback_analysis_df['concise_theme'] = mapped_results.apply(lambda x: x[0] if isinstance(x, tuple) else 'Error')
    feedback_analysis_df['sentiment'] = mapped_results.apply(lambda x: x[1] if isinstance(x, tuple) else 'Error')
    
    # Save output
    output_file = os.path.join(OUTPUT_DIRECTORY, "themes_tagged_feedback.csv")
    feedback_analysis_df.to_csv(output_file, index=False)
    print(f"✓ Saved themes to: {output_file}")
    
    # Display statistics
    print(f"\n Theme extraction complete:")
    print(f"  - Total feedback processed: {len(feedback_analysis_df)}")
    print(f"  - Unique themes identified: {feedback_analysis_df['concise_theme'].nunique()}")
    print(f"  - Sentiment distribution:")
    print(feedback_analysis_df['sentiment'].value_counts().to_string(header=False))
    
    return feedback_analysis_df

# ========================================
# Theme Consolidation Functions
# ========================================

def consolidate_themes(feedback_analysis_df, model):
    """Consolidate themes into 5 major categories"""
    print("\n" + "=" * 60)
    print("THEME CONSOLIDATION")
    print("=" * 60)
    
    valid_themes_df = feedback_analysis_df[~feedback_analysis_df['concise_theme'].str.contains("Error", na=False)]
    theme_list_for_prompt = valid_themes_df['concise_theme'].unique().tolist()
    
    print(f"Consolidating {len(theme_list_for_prompt)} unique themes into 5 categories...")
    
    try:
        consolidation_prompt = THEME_CONSOLIDATION_PROMPT.format(
            theme_list=json.dumps(theme_list_for_prompt)
        )
        
        print("Sending request to LLM...")
        consolidation_response = model.generate_content(consolidation_prompt)
        cleaned_response = (
            consolidation_response.text.strip()
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        
        consolidated_data = json.loads(cleaned_response)
        
        # Save consolidation results
        output_file = os.path.join(OUTPUT_DIRECTORY, "theme_consolidation.json")
        with open(output_file, 'w') as f:
            json.dump(consolidated_data, f, indent=2)
        print(f"✓ Saved consolidation to: {output_file}")
        
        # Create mapping
        theme_to_category_map = {}
        for category in consolidated_data.get("theme_categories", []):
            category_name = category.get("category_name")
            included_themes = category.get("included_themes", [])
            if category_name:
                for theme in included_themes:
                    theme_to_category_map[theme] = category_name
        
        # Apply mapping
        feedback_analysis_df['theme_category'] = feedback_analysis_df['concise_theme'].map(theme_to_category_map)
        
        print("\nCategory distribution:")
        print(feedback_analysis_df['theme_category'].value_counts().to_string())
        
        # Save the complete DataFrame with theme categories
        output_file = os.path.join(OUTPUT_DIRECTORY, "themes_consolidated_feedback.csv")
        feedback_analysis_df.to_csv(output_file, index=False)
        print(f"✓ Saved consolidated feedback with categories to: {output_file}")
        
        return feedback_analysis_df, consolidated_data
        
    except Exception as e:
        print(f"ERROR: Failed to consolidate themes: {e}")
        return feedback_analysis_df, None

# ========================================
# Trend Analysis Functions
# ========================================

def analyze_trends(feedback_analysis_df, model):
    """Analyze weekly and monthly trends"""
    print("\n" + "=" * 60)
    print("TREND ANALYSIS")
    print("=" * 60)
    
    plot_df = feedback_analysis_df.copy()
    plot_df['created_at'] = pd.to_datetime(plot_df['created_at'])
    plot_df.set_index('created_at', inplace=True)
    
    main_themes = [theme for theme in plot_df['theme_category'].unique() 
                   if theme not in ['Other', 'Error', None]]
    plot_df_filtered = plot_df[plot_df['theme_category'].isin(main_themes)]
    
    # Weekly trends
    print("Calculating weekly trends...")
    weekly_trends = plot_df_filtered.groupby([
        pd.Grouper(freq='W'), 
        'theme_category'
    ]).size().unstack('theme_category').fillna(0)
    
    # Save weekly trends
    output_file = os.path.join(OUTPUT_DIRECTORY, "trend_weekly_volume.csv")
    weekly_trends.to_csv(output_file)
    print(f"✓ Saved weekly trends to: {output_file}")
    
    # Monthly trends
    print("Calculating monthly trends...")
    monthly_trends = plot_df_filtered.groupby([
        pd.Grouper(freq='ME'), 
        'theme_category'
    ]).size().unstack('theme_category').fillna(0)
    
    # Save monthly trends
    output_file = os.path.join(OUTPUT_DIRECTORY, "trend_monthly_volume.csv")
    monthly_trends.to_csv(output_file)
    print(f"✓ Saved monthly trends to: {output_file}")
    
    # Generate visualization
    print("Generating trend visualization...")
    sns.set_style("ticks")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for theme in weekly_trends.columns:
        ax.plot(weekly_trends.index, weekly_trends[theme], linestyle='-', label=theme, linewidth=2)
    
    ax.set_title('Weekly Volume of Customer Feedback by Theme', fontsize=16, weight='bold')
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Number of Feedback Messages', fontsize=12)
    ax.grid(False)
    ax.legend(title='Theme Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_DIRECTORY, "trend_weekly_chart.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved trend chart to: {output_file}")
    plt.close()
    
    # Generate summary
    print("Generating AI-powered trend summary...")
    weekly_data_str = weekly_trends.to_string()
    monthly_data_str = monthly_trends.to_string()
    
    summary_prompt = TREND_SUMMARY_PROMPT.format(
        weekly_data=weekly_data_str,
        monthly_data=monthly_data_str
    )
    
    try:
        summary_response = model.generate_content(summary_prompt)
        summary_text = summary_response.text
        
        # Save as text
        output_file = os.path.join(OUTPUT_DIRECTORY, "trend_summary.txt")
        with open(output_file, 'w') as f:
            f.write(summary_text)
        print(f"✓ Saved trend summary to: {output_file}")
        
        # Save as JSON
        summary_json = {"summary": summary_text, "generated_at": datetime.now().isoformat()}
        output_file = os.path.join(OUTPUT_DIRECTORY, "trend_summary.json")
        with open(output_file, 'w') as f:
            json.dump(summary_json, f, indent=2)
        print(f"✓ Saved trend summary JSON to: {output_file}")
        
        return weekly_trends, monthly_trends
        
    except Exception as e:
        print(f"ERROR: Failed to generate summary: {e}")
        return weekly_trends, monthly_trends

# ========================================
# Anomaly Detection Functions
# ========================================

def detect_anomalies_comprehensive(feedback_analysis_df):
    """Comprehensive anomaly detection using multiple methods"""
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION")
    print("=" * 60)
    
    analysis_df = feedback_analysis_df.copy()
    analysis_df['created_at'] = pd.to_datetime(analysis_df['created_at'])
    analysis_df.set_index('created_at', inplace=True)
    
    main_themes = [theme for theme in analysis_df['theme_category'].unique() 
                   if theme not in ['Other', 'Error', None]]
    analysis_df = analysis_df[analysis_df['theme_category'].isin(main_themes)]
    
    print(f"Analyzing anomalies across {len(main_themes)} themes...")
    
    # Prepare data
    negative_feedback_df = analysis_df[analysis_df['sentiment'] == 'Negative']
    weekly_negative_trends = negative_feedback_df.groupby([
        pd.Grouper(freq='W'), 
        'theme_category'
    ]).size().unstack('theme_category').fillna(0)
    
    weekly_total_trends = analysis_df.groupby([
        pd.Grouper(freq='W'),
        'theme_category'
    ]).size().unstack('theme_category').fillna(0)
    
    weekly_negative_ratio = weekly_negative_trends.div(weekly_total_trends.replace(0, np.nan))
    
    # Detection functions
    def detect_anomalies_iqr(data, multiplier=1.5):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + multiplier * IQR
        return data[data > upper_bound], upper_bound
    
    def detect_anomalies_mad(data, threshold=3.0):
        median = data.median()
        mad = np.median(np.abs(data - median))
        if mad == 0:
            mad = np.mean(np.abs(data - median))
        if mad > 0:
            modified_z_scores = 0.6745 * (data - median) / mad
            return data[modified_z_scores > threshold], median + threshold * mad / 0.6745
        return pd.Series(), np.nan
    
    def detect_anomalies_zscore(data, threshold=2.0):
        mean = data.mean()
        std = data.std()
        if std > 0:
            z_scores = (data - mean) / std
            upper_threshold = mean + threshold * std
            return data[z_scores > threshold], upper_threshold
        return pd.Series(), np.nan
    
    # Detect anomalies
    anomalies = []
    MIN_WEEKS = 6
    MIN_VARIATION_CV = 0.2
    
    with tqdm(total=len(weekly_negative_trends.columns), desc="Detecting anomalies") as pbar:
        for theme in weekly_negative_trends.columns:
            theme_data = weekly_negative_trends[theme]
            theme_total = weekly_total_trends[theme]
            theme_ratio = weekly_negative_ratio[theme].dropna()
            
            pbar.update(1)
            
            if len(theme_data[theme_data > 0]) < MIN_WEEKS:
                continue
            
            if theme_data.mean() > 0:
                cv = theme_data.std() / theme_data.mean()
                if cv < MIN_VARIATION_CV:
                    continue
            
            # Apply multiple detection methods
            iqr_anomalies, iqr_threshold = detect_anomalies_iqr(theme_data, multiplier=1.5)
            mad_anomalies, mad_threshold = detect_anomalies_mad(theme_data, threshold=2.5)
            zscore_anomalies, zscore_threshold = detect_anomalies_zscore(theme_data, threshold=1.8)
            ratio_anomalies, ratio_threshold = detect_anomalies_iqr(theme_ratio, multiplier=1.5)
            
            # Combine results
            all_weeks = set()
            all_weeks.update(iqr_anomalies.index)
            all_weeks.update(mad_anomalies.index)
            all_weeks.update(zscore_anomalies.index)
            
            for week in all_weeks:
                detection_count = 0
                methods_detected = []
                
                if week in iqr_anomalies.index:
                    detection_count += 1
                    methods_detected.append("IQR")
                if week in mad_anomalies.index:
                    detection_count += 1
                    methods_detected.append("MAD")
                if week in zscore_anomalies.index:
                    detection_count += 1
                    methods_detected.append("Z-score")
                if week in ratio_anomalies.index:
                    detection_count += 1
                    methods_detected.append("Ratio")
                
                confidence = detection_count / 4.0
                
                if detection_count >= 2 or (week in ratio_anomalies.index and theme_ratio[week] > 0.8):
                    severity = (theme_data[week] - theme_data.mean()) / (theme_data.std() + 0.001)
                    
                    anomalies.append({
                        "theme": theme,
                        "week_timestamp": week.isoformat(),
                        "count": int(theme_data[week]),
                        "total_feedback": int(theme_total[week]),
                        "negative_ratio": float(theme_ratio[week]) if week in theme_ratio.index else None,
                        "mean_count": round(float(theme_data.mean()), 1),
                        "methods_detected": methods_detected,
                        "confidence": round(confidence, 2),
                        "severity": round(float(severity), 2),
                        "iqr_threshold": round(float(iqr_threshold), 1) if not pd.isna(iqr_threshold) else None
                    })
    
    anomalies = sorted(anomalies, key=lambda x: x['severity'], reverse=True)
    
    print(f"✓ Detected {len(anomalies)} anomalies")
    
    # Save anomaly report
    output_file = os.path.join(OUTPUT_DIRECTORY, "anomaly_report.json")
    with open(output_file, 'w') as f:
        json.dump(anomalies, f, indent=2)
    print(f"✓ Saved anomaly report to: {output_file}")
    
    # Generate visualization with anomalies
    print("Generating anomaly visualization...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: Total trends with anomalies
    for theme in weekly_total_trends.columns:
        ax1.plot(weekly_total_trends.index, weekly_total_trends[theme], 
                linestyle='-', alpha=0.7, label=theme, linewidth=2)
    
    # Add anomaly markers
    for anomaly in anomalies:
        week = pd.to_datetime(anomaly['week_timestamp'])
        theme = anomaly['theme']
        if week in weekly_total_trends.index and theme in weekly_total_trends.columns:
            ax1.axvline(x=week, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    ax1.set_title('Weekly Feedback Volume with Detected Anomalies', fontsize=14, weight='bold')
    ax1.set_xlabel('Week', fontsize=11)
    ax1.set_ylabel('Total Feedback Count', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)
    
    # Plot 2: Negative sentiment ratio
    for theme in weekly_negative_ratio.columns:
        ratio_data = weekly_negative_ratio[theme].fillna(0) * 100
        ax2.plot(weekly_negative_ratio.index, ratio_data, 
                linestyle='-', marker='o', markersize=4, alpha=0.7, label=theme)
    
    ax2.set_title('Negative Sentiment Ratio Over Time', fontsize=14, weight='bold')
    ax2.set_xlabel('Week', fontsize=11)
    ax2.set_ylabel('Negative Feedback %', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=9)
    
    plt.suptitle(f'Anomaly Detection Dashboard - {len(anomalies)} Anomalies Detected', 
                fontsize=16, weight='bold', y=0.99)
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_DIRECTORY, "trend_weekly_chart_with_anomalies.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved anomaly chart to: {output_file}")
    plt.close()
    
    # Save negative sentiment ratio
    output_file = os.path.join(OUTPUT_DIRECTORY, "negative_sentiment_ratio.csv")
    weekly_negative_ratio.to_csv(output_file)
    print(f"✓ Saved negative sentiment ratio to: {output_file}")
    
    # Statistical summary
    print("Generating statistical summary...")
    summary_stats = pd.DataFrame({
        'Theme': weekly_negative_trends.columns,
        'Mean': weekly_negative_trends.mean(),
        'Std Dev': weekly_negative_trends.std(),
        'CV': weekly_negative_trends.std() / weekly_negative_trends.mean(),
        'Max': weekly_negative_trends.max(),
        'Weeks >0': (weekly_negative_trends > 0).sum()
    })
    summary_stats = summary_stats.round(2).sort_values('Mean', ascending=False)
    
    output_file = os.path.join(OUTPUT_DIRECTORY, "statistical_summary.csv")
    summary_stats.to_csv(output_file, index=False)
    print(f"✓ Saved statistical summary to: {output_file}")
    
    return anomalies, weekly_negative_ratio, summary_stats

# ========================================
# Segmentation Functions
# ========================================

def perform_segmentation(feedback_analysis_df, product_usage_df):
    """Perform user segmentation analysis"""
    print("\n" + "=" * 60)
    print("USER SEGMENTATION")
    print("=" * 60)
    
    print("Merging feedback and usage data...")
    merged_df = pd.merge(feedback_analysis_df, product_usage_df, on='customer_id', how='inner')
    print(f"✓ Merged {len(merged_df)} records")
    
    # Save merged data
    output_file = os.path.join(OUTPUT_DIRECTORY, "feedback_usage_merged.parquet")
    merged_df.to_parquet(output_file)
    print(f"✓ Saved merged data to: {output_file}")
    
    analysis_df = merged_df.copy()
    
    # Segmentation by subscription tier
    print("\nAnalyzing by subscription tier...")
    tier_analysis = pd.crosstab(analysis_df['theme_category'], analysis_df['subscription_tier'])
    
    output_file = os.path.join(OUTPUT_DIRECTORY, "segmentation_by_tier.csv")
    tier_analysis.to_csv(output_file)
    print(f"✓ Saved tier segmentation to: {output_file}")
    
    # Segmentation by spend cohort
    print("Creating spend cohorts...")
    try:
        analysis_df['spend_cohort'] = pd.qcut(
            analysis_df['total_spend'],
            q=3,
            labels=['Low Spender', 'Mid Spender', 'High Spender']
        )
    except ValueError:
        analysis_df['spend_cohort'] = pd.qcut(
            analysis_df['total_spend'],
            q=2,
            labels=['Low Spender', 'High Spender'],
            duplicates='drop'
        )
    
    spend_analysis = pd.crosstab(analysis_df['theme_category'], analysis_df['spend_cohort'])
    
    output_file = os.path.join(OUTPUT_DIRECTORY, "segmentation_by_spend.csv")
    spend_analysis.to_csv(output_file)
    print(f"✓ Saved spend segmentation to: {output_file}")
    
    # Segmentation by approval ratio
    print("Creating approval ratio cohorts...")
    analysis_df['approval_ratio'] = (
        analysis_df['advance_approvals_30d'] / analysis_df['advance_attempts_30d']
    ).fillna(0)
    
    bins = [-0.1, 0.1, 0.5, 1.01]
    labels = ['Low (0%)', 'Mid (1-50%)', 'High (>50%)']
    analysis_df['approval_ratio_cohort'] = pd.cut(
        analysis_df['approval_ratio'], 
        bins=bins, 
        labels=labels, 
        right=True
    )
    
    approval_analysis = pd.crosstab(analysis_df['theme_category'], analysis_df['approval_ratio_cohort'])
    
    output_file = os.path.join(OUTPUT_DIRECTORY, "segmentation_by_approval_ratio.csv")
    approval_analysis.to_csv(output_file)
    print(f"✓ Saved approval ratio segmentation to: {output_file}")
    
    print("\nSegmentation complete:")
    print(f"  - Tiers analyzed: {list(tier_analysis.columns)}")
    print(f"  - Spend cohorts: {list(spend_analysis.columns)}")
    print(f"  - Approval cohorts: {list(approval_analysis.columns)}")
    
    return analysis_df, tier_analysis, spend_analysis, approval_analysis

# ========================================
# Recommendation Generation
# ========================================

def generate_recommendations(feedback_analysis_df, anomalies, tier_analysis, 
                            spend_analysis, approval_analysis, model):
    """Generate strategic recommendations based on analysis"""
    print("\n" + "=" * 60)
    print("GENERATING RECOMMENDATIONS")
    print("=" * 60)
    
    def prepare_llm_context(top_n_themes=3):
        """Prepare context for LLM"""
        top_themes = feedback_analysis_df['theme_category'].value_counts().nlargest(top_n_themes).index.tolist()
        top_themes_str = ", ".join(f"'{theme}'" for theme in top_themes)
        
        anomaly_str = ""
        if anomalies:
            for anomaly in anomalies[:3]:  # Top 3 anomalies
                week_str = pd.to_datetime(anomaly['week_timestamp']).strftime('%Y-%m-%d')
                anomaly_str += (
                    f"- A significant spike in negative feedback for '{anomaly['theme']}' "
                    f"occurred during the week of {week_str}.\n"
                )
        else:
            anomaly_str = "No significant anomalies in negative feedback were detected.\n"
        
        segment_insights_str = ""
        for theme in top_themes:
            if theme in tier_analysis.index:
                worst_tier = tier_analysis.loc[theme].idxmax()
                segment_insights_str += (
                    f"- For the theme '{theme}', the most affected subscription tier is '{worst_tier}'.\n"
                )
            if theme in spend_analysis.index:
                worst_spend = spend_analysis.loc[theme].idxmax()
                segment_insights_str += (
                    f"- The user group most impacted by '{theme}' is our '{worst_spend}' cohort.\n"
                )
            if theme in approval_analysis.index:
                worst_approval = approval_analysis.loc[theme].idxmax()
                segment_insights_str += (
                    f"- '{theme}' issues are most prevalent among users in the '{worst_approval}' approval ratio cohort.\n"
                )
        
        full_context = f"""
[Data Dossier: Key Analytical Findings]

1.  **Top Recurring Feedback Themes**: The most frequent customer issues are: {top_themes_str}.

2.  **Trend & Anomaly Report**:
{anomaly_str}
3.  **User Segmentation Insights**:
{segment_insights_str}
"""
        return full_context
    
    print("Preparing context for recommendations...")
    data_dossier = prepare_llm_context()
    
    generation_config = genai.types.GenerationConfig(temperature=0.1)
    
    final_prompt = RECOMMENDATION_PROMPT.format(data_dossier=data_dossier)
    
    try:
        print("Sending request to LLM for strategic analysis...")
        response = model.generate_content(
            final_prompt,
            generation_config=generation_config
        )
        
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        recommendations_data = json.loads(cleaned_response)
        
        # Save recommendations
        output_file = os.path.join(OUTPUT_DIRECTORY, "strategic_recommendations.json")
        with open(output_file, 'w') as f:
            json.dump(recommendations_data, f, indent=2)
        print(f"✓ Saved recommendations to: {output_file}")
        
        # Display recommendations
        print("\n" + "-" * 40)
        print("STRATEGIC RECOMMENDATIONS")
        print("-" * 40)
        for rec in recommendations_data.get("recommendations", []):
            print(f"\nPriority {rec.get('priority')}: {rec.get('recommendation_title')}")
            print(f"Rationale: {rec.get('rationale')}")
            print(f"Expected Outcome: {rec.get('business_outcome')}")
        
        return recommendations_data
        
    except Exception as e:
        print(f"ERROR: Failed to generate recommendations: {e}")
        return None

# ========================================
# Explainability Functions
# ========================================

def generate_explainability_samples(feedback_analysis_df):
    """Generate sample feedback for each theme category"""
    print("\n" + "=" * 60)
    print("EXPLAINABILITY SAMPLES")
    print("=" * 60)
    
    print("Generating explanatory samples for each theme...")
    
    unique_themes = [theme for theme in feedback_analysis_df['theme_category'].unique() 
                    if theme not in [None, 'Error']][:5]  # Top 5 themes
    
    explainability_data = {}
    
    for theme in unique_themes:
        theme_df = feedback_analysis_df[feedback_analysis_df['theme_category'] == theme]
        samples = theme_df.sample(n=min(len(theme_df), 5))  # 5 examples per theme
        
        explainability_data[theme] = samples[['message', 'sentiment', 'concise_theme']].to_dict('records')
    
    # Save as JSON
    output_file = os.path.join(OUTPUT_DIRECTORY, "explainability_samples.json")
    with open(output_file, 'w') as f:
        json.dump(explainability_data, f, indent=2)
    print(f"✓ Saved explainability samples to: {output_file}")
    
    # Save as formatted HTML
    html_content = "<html><head><style>"
    html_content += "table {border-collapse: collapse; margin: 20px;}"
    html_content += "th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}"
    html_content += "th {background-color: #f2f2f2;}"
    html_content += "</style></head><body>"
    html_content += "<h1>Explainability Samples</h1>"
    
    for theme, samples in explainability_data.items():
        html_content += f"<h2>{theme}</h2>"
        html_content += "<table>"
        html_content += "<tr><th>Message</th><th>Sentiment</th><th>Specific Theme</th></tr>"
        for sample in samples:
            html_content += "<tr>"
            html_content += f"<td>{sample['message']}</td>"
            html_content += f"<td>{sample['sentiment']}</td>"
            html_content += f"<td>{sample['concise_theme']}</td>"
            html_content += "</tr>"
        html_content += "</table>"
    
    html_content += "</body></html>"
    
    output_file = os.path.join(OUTPUT_DIRECTORY, "explainability_samples.html")
    with open(output_file, 'w') as f:
        f.write(html_content)
    print(f"✓ Saved explainability HTML to: {output_file}")
    
    print(f"\nGenerated samples for {len(unique_themes)} themes")
    for theme in unique_themes:
        print(f"  - {theme}: {len(explainability_data[theme])} samples")
    
    return explainability_data

# ========================================
# Main Pipeline
# ========================================

def main():
    """Main analysis pipeline"""
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print(" CUSTOMER FEEDBACK ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize
        model = initialize_llm()
        
        # Load data
        feedback_df, product_usage_df = load_data()
        
        # 1. Theme Extraction
        feedback_analysis_df = extract_themes(feedback_df, model)
        
        # 2. Theme Consolidation
        feedback_analysis_df, consolidated_data = consolidate_themes(feedback_analysis_df, model)
        
        # 3. Trend Analysis
        weekly_trends, monthly_trends = analyze_trends(feedback_analysis_df, model)
        
        # 4. Anomaly Detection
        anomalies, negative_ratio, stats_summary = detect_anomalies_comprehensive(feedback_analysis_df)
        
        # 5. User Segmentation
        analysis_df, tier_analysis, spend_analysis, approval_analysis = perform_segmentation(
            feedback_analysis_df, product_usage_df
        )
        
        # 6. Generate Recommendations
        recommendations = generate_recommendations(
            feedback_analysis_df, anomalies, tier_analysis, 
            spend_analysis, approval_analysis, model
        )
        
        # 7. Generate Explainability Samples
        explainability_data = generate_explainability_samples(feedback_analysis_df)
        
        # Summary
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(" ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        print(f"All outputs saved to: {OUTPUT_DIRECTORY}")
        
        # List all generated files
        print("\nGenerated files:")
        for file in sorted(os.listdir(OUTPUT_DIRECTORY)):
            file_path = os.path.join(OUTPUT_DIRECTORY, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / 1024  # KB
                print(f"  - {file} ({size:.1f} KB)")
        
    except Exception as e:
        print(f"\n ERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())