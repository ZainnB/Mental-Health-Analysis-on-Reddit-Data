import praw
import pandas as pd
import time
from datetime import datetime
import os

# === Initialize Reddit API ===
reddit = praw.Reddit(
    client_id='-',
    client_secret='-',
    user_agent='-'
)

# === Config ===
subreddits = [
    'mentalhealth', 'depression', 'anxiety', 'selfimprovement', 'therapy',
    'socialanxiety', 'bipolar', 'OCD', 'ptsd', 'addiction', 'SuicideWatch',
    'stress', 'lonely', 'panicattack', 'psychotherapy', 'mentalillness'
]

posts_per_query = 300
save_every = 300
output_file = "reddit_mental_health_posts.xlsx"

# === Step 1: Load Existing IDs ===
seen_ids = set()
if os.path.exists(output_file):
    try:
        old_df = pd.read_excel(output_file)
        seen_ids = set(old_df['id'].astype(str).tolist())
        print(f"📄 Loaded {len(seen_ids)} existing post IDs.")
    except Exception as e:
        print(f"⚠️ Failed to load existing file: {e}")

# === Step 2: Scrape Posts ===
all_posts = []

def save_to_excel(posts, path):
    df = pd.DataFrame(posts)
    df.drop_duplicates(subset="id", inplace=True)
    
    # If file exists, append to old data before saving
    if os.path.exists(path):
        old_df = pd.read_excel(path)
        df = pd.concat([old_df, df], ignore_index=True)
        df.drop_duplicates(subset="id", inplace=True)
    
    df.to_excel(path, index=False)
    print(f"\n💾 Saved {len(df)} total posts to {path}")

for subreddit_name in subreddits:
    print(f"\n🔎 Fetching from r/{subreddit_name}...")
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.search(query="*", sort="top", time_filter="all", limit=posts_per_query):
            post_id = submission.id
            if post_id in seen_ids:
                continue  # Skip if already seen

            post = {
                'id': post_id,
                'subreddit': subreddit_name,
                'title': submission.title,
                'text': submission.selftext,
                'score': submission.score,
                'created_utc': datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                'num_comments': submission.num_comments,
                'url': submission.url
            }
            all_posts.append(post)
            seen_ids.add(post_id)

            print(f"✔️ {post['created_utc']} | {post['title'][:60]}...")

            if len(all_posts) % save_every == 0:
                save_to_excel(all_posts, output_file)

            time.sleep(1)  # Respect rate limits
    except Exception as e:
        print(f"⚠️ Error fetching from r/{subreddit_name}: {e}")
        continue

# === Final Save ===
if all_posts:
    save_to_excel(all_posts, output_file)
    print(f"\n✅ Final save complete. New posts added: {len(all_posts)}")
else:
    print("\n⚠️ No new data collected.")
