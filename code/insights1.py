import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
from wordcloud import WordCloud
import os

# === Step 1: Load Data ===
file_path = 'reddit_with_sentiment_emotion.xlsx'  # Path to your master .xlsx file
df = pd.read_excel(file_path)

# Convert to datetime and extract year, month, and other relevant time features
df['created_utc'] = pd.to_datetime(df['created_utc'])
df['year'] = df['created_utc'].dt.year
df['month'] = df['created_utc'].dt.month
df['year_month'] = df['created_utc'].dt.to_period('M')

# === Step 2: Output Directory ===
output_dir = "/mnt/data/reddit_analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

# === Section 1: Sentiment & Emotion Insights ===

# 1. Monthly Sentiment Trend (Line chart)
def plot_monthly_sentiment_trend(df):
    monthly_counts = df.groupby(['year_month']).size()
    sentiment_monthly = df.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)
    normalized_sentiment = sentiment_monthly.div(monthly_counts, axis=0)
    
    normalized_sentiment.plot(kind='line', marker='o', figsize=(18, 9), linewidth=2)
    plt.title('Normalized Monthly Sentiment Trend')
    plt.xlabel('Year-Month')
    plt.ylabel('Proportion of Posts')
    plt.legend(title='Sentiment')
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'normalized_monthly_sentiment_trend.png'))
    plt.show()

# 2. Monthly Emotion Trend (Line chart)
def plot_monthly_emotion_trend(df):
    monthly_counts = df.groupby(['year_month']).size()
    emotion_monthly = df.groupby(['year_month', 'emotion']).size().unstack(fill_value=0)
    normalized_emotion = emotion_monthly.div(monthly_counts, axis=0)
    
    normalized_emotion.plot(kind='line', marker='o', figsize=(18, 9), linewidth=2)
    plt.title('Normalized Monthly Emotion Trend')
    plt.xlabel('Year-Month')
    plt.ylabel('Proportion of Posts')
    plt.legend(title='Emotion')
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'normalized_monthly_emotion_trend.png'))
    plt.show()

# 3. Pre- vs. Post-COVID Sentiment Distribution (Pie chart)
def plot_pre_post_covid_sentiment(df):
    pre_covid = df[df['year'] < 2020]['sentiment'].value_counts(normalize=True)
    post_covid = df[df['year'] >= 2020]['sentiment'].value_counts(normalize=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pre_covid.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=axes[0])
    axes[0].set_ylabel('')
    axes[0].set_title('Sentiment Distribution Pre-COVID (Before 2020)')
    post_covid.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=axes[1])
    axes[1].set_ylabel('')
    axes[1].set_title('Sentiment Distribution Post-COVID (2020 Onwards)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pre_post_covid_sentiment.png'))
    plt.show()

# 4. Pre- vs. Post-COVID Emotion Distribution (Pie chart)
def plot_pre_post_covid_emotion(df):
    pre_covid = df[df['year'] < 2020]['emotion'].value_counts(normalize=True)
    post_covid = df[df['year'] >= 2020]['emotion'].value_counts(normalize=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pre_covid.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=axes[0])
    axes[0].set_ylabel('')
    axes[0].set_title('Emotion Distribution Pre-COVID (Before 2020)')
    post_covid.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=axes[1])
    axes[1].set_ylabel('')
    axes[1].set_title('Emotion Distribution Post-COVID (2020 Onwards)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pre_post_covid_emotion.png'))
    plt.show()

# 5. Emotion vs. Sentiment Cross-Tab (Heatmap)
def plot_emotion_sentiment_cross_tab(df):
    cross_tab = pd.crosstab(df['sentiment'], df['emotion'], normalize='index')
    plt.figure(figsize=(10, 6))
    sns.heatmap(cross_tab, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title('Emotion vs. Sentiment Cross-Tab')
    plt.ylabel('Sentiment')
    plt.xlabel('Emotion')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_sentiment_cross_tab.png'))
    plt.show()

# === Section 2: User Activity & Engagement ===

# 6. Monthly Post Volume (Bar chart)
def plot_monthly_post_volume(df):
    post_volume = df.groupby('year_month').size()
    normalized_post_volume = post_volume / post_volume.sum()

    normalized_post_volume.plot(kind='bar', figsize=(20, 7), color='skyblue')
    plt.title('Normalized Monthly Post Volume')
    plt.xlabel('Year-Month')
    plt.ylabel('Proportion of Posts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normalized_monthly_post_volume.png'))
    plt.show()

# 7. Average Score / Comments Over Time (Line chart)
def plot_avg_score_comments(df):
    avg_score_comments = df.groupby('year_month').agg({'score': 'mean', 'num_comments': 'mean'})
    avg_score_comments.plot(kind='line', figsize=(14, 7), linewidth=2)
    plt.title('Average Score & Comments Over Time')
    plt.xlabel('Year-Month')
    plt.ylabel('Average Value')
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'avg_score_comments.png'))
    plt.show()

# === Section 3: Subreddit-Based Analysis ===

# 8. Sentiment Distribution by Subreddit (Bar chart)
def plot_sentiment_by_subreddit(df):
    sentiment_by_subreddit = df.groupby('subreddit')['sentiment'].value_counts(normalize=True).unstack().fillna(0)
    sentiment_by_subreddit.plot(kind='bar', stacked=True, figsize=(14, 7))
    plt.title('Sentiment Distribution by Subreddit')
    plt.xlabel('Subreddit')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_by_subreddit.png'))
    plt.show()

# 9. Emotion Distribution by Subreddit (Bar chart)
def plot_emotion_by_subreddit(df):
    emotion_by_subreddit = df.groupby('subreddit')['emotion'].value_counts(normalize=True).unstack().fillna(0)
    emotion_by_subreddit.plot(kind='bar', stacked=True, figsize=(14, 7))
    plt.title('Emotion Distribution by Subreddit')
    plt.xlabel('Subreddit')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_by_subreddit.png'))
    plt.show()

# === Final Step: Generate All Insights ===
def generate_all_insights(df):
    plot_monthly_sentiment_trend(df)
    plot_monthly_emotion_trend(df)
    plot_pre_post_covid_sentiment(df)
    plot_pre_post_covid_emotion(df)
    plot_emotion_sentiment_cross_tab(df)
    plot_monthly_post_volume(df)
    plot_avg_score_comments(df)
    plot_sentiment_by_subreddit(df)
    plot_emotion_by_subreddit(df)

# Run the analysis and generate insights
generate_all_insights(df)

# Notify the user that the results have been saved
output_dir = "/reddit_analysis_outputs"
