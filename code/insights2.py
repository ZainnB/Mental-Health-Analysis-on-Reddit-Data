import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from collections import Counter
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

# === Section 5: Correlation Insights ===

# 12. Sentiment vs. Score (Box plot or Scatter plot)
def plot_sentiment_vs_score(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='sentiment', y='score', data=df)
    plt.title('Sentiment vs. Score')
    plt.xlabel('Sentiment')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_vs_score.png'))
    plt.show()

# 13. Emotion vs. Comment Count (Box plot)
def plot_emotion_vs_comment_count(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='emotion', y='num_comments', data=df)
    plt.title('Emotion vs. Comment Count')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Comments')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_vs_comment_count.png'))
    plt.show()

# === Section 6: Advanced / Optional ===

# 14. Word Clouds by Emotion or Sentiment
def plot_wordcloud_by_emotion(df, emotion_column='emotion'):
    emotions = df[emotion_column].unique()
    for emotion in emotions:
        text = " ".join(df[df[emotion_column] == emotion]['clean_text'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {emotion} Emotion')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'wordcloud_{emotion}.png'))
        plt.show()

# 15. Topic Modeling (LDA or BERTopic) Per Emotion
def plot_lda_topics_by_emotion(df, emotion_column='emotion', num_topics=5):
    emotions = df[emotion_column].unique()
    
    # Prepare text data for LDA
    cv = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
    
    for emotion in emotions:
        text_data = df[df[emotion_column] == emotion]['clean_text'].dropna()
        X = cv.fit_transform(text_data)
        
        # Apply LDA
        lda = LDA(n_components=num_topics, random_state=42)
        lda.fit(X)
        
        # Display the topics with top words
        print(f"Top topics for {emotion} Emotion:")
        for topic_idx, topic in enumerate(lda.components_):
            print(f"Topic {topic_idx + 1}:")
            print(" ".join([cv.get_feature_names_out()[i] for i in topic.argsort()[:-10 - 1:-1]]))
        print("\n" + "="*80)
        
        # Create a word cloud for each topic
        for topic_idx, topic in enumerate(lda.components_):
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
                dict(zip(cv.get_feature_names_out(), topic))
            )
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Topic {topic_idx + 1} for {emotion} Emotion')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'lda_topic_{emotion}_{topic_idx + 1}.png'))
            plt.show()

# === Final Step: Generate All Insights for Sections 5 and 6 ===
def generate_section5_and_section6_insights(df):
    # Correlation Insights (Section 5)
    plot_sentiment_vs_score(df)
    plot_emotion_vs_comment_count(df)
    
    # Advanced Insights (Section 6)
    plot_wordcloud_by_emotion(df)
    plot_lda_topics_by_emotion(df)

# Run the analysis and generate insights for Sections 5 and 6
generate_section5_and_section6_insights(df)

# Notify the user that the results have been saved
output_dir
