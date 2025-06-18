# Reddit Mental Health Sentiment & Emotion Analysis

This project analyzes Reddit mental health discussions over time using sentiment and emotion analysis. It focuses on uncovering emotional and psychological trends across communities like r/depression and r/Anxiety using time-normalized metrics.

---

## 📌 Objective

To track how sentiment and emotion patterns in mental health subreddits have evolved—especially across critical periods like the COVID-19 pandemic—by analyzing user-generated content and normalizing it by post frequency over time.

---

## 📂 Project Structure

📁 code/ → Contains all code .py files 
    📄 data_reddit.py → Calls PRAW Reddit api to fetch and store posts data
    📄 analyze.py → Preprocesses data and performs sentiment and emotion tagging (also saves it in new CSV)
    📄 insights1.py → Generates meaningful time-normalized graphs of both sentiment and emotional tags 
    📄 insights2.py → Generates relations between sentiment and emotion and wordclouds.
📁 data/ → Contains both 
    📄 initial collected Reddit posts data and
    📄 processed and normalized data in CSV format
📁 visuals/ → Stores output graphs and visualizations
📁 doc/ → Contains all doumentation of project
    📄 research_prposal.pdf → Initial project proposal
    📄 progress_report.pdf → Contains midway progress report
    📄 research_paper.pdf → Complete project report and analysis
    📄 project_presentation.pdf → Complete and organized presentation covering all aspects of project


---

## ⚙️ Technologies & Tools

- **Language**: Python  
- **APIs**: PRAW (Reddit API)  
- **Libraries**: `transformers`, `pandas`, `numpy`, `nltk`, `matplotlib`, `seaborn`  
- **Models**:  
  - **RoBERTa-base** for contextual sentiment classification  
  - **Hartmann’s Emotion Lexicon** for multi-label emotion tagging

---

## 🔄 Workflow

1. **Data Collection**
   - Extract Reddit posts via PRAW from subreddits like r/depression, r/Anxiety

2. **Preprocessing**
   - Lowercasing, punctuation removal, stopword filtering, lemmatization

3. **Sentiment Analysis**
   - Using RoBERTa transformer model for fine-grained sentiment labels

4. **Emotion Detection**
   - Lexicon-based multi-emotion tagging using Hartmann's word-emotion associations

5. **Normalization**
   - Group and normalize sentiment/emotion scores per month by post count

6. **Visualization**
   - Plot trends over time using seaborn and matplotlib
---

## 📊 Output

- Monthly trends in sentiment and emotion
- CSV files for raw and normalized scores
- Graphs of psychological patterns in mental health communities over time

---

## 📁 Deliverables

- ✅ Cleaned and processed dataset  
- ✅ Python codebase (modular and documented)  
- ✅ Visualizations (PNG/JPG graphs)  
- ✅ Research paper summarizing methodology, results, and insights  

---

## 🙋‍♂️ Author

**Zain Baig**  
BSCS Student | Python Developer | Data Enthusiast  
🔗 [LinkedIn](https://www.linkedin.com/in/zain-baig-04790b260/) | ✉️ zainbaig@example.com

---

## 📄 License

This project is for educational and non-commercial research use only.
