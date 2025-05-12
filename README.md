# 📰✨ GENai_Project_22BEC10034

🚀 **Author:** Harsh Karekar (22BEC10034)  
🎓 **Project Title:** Fake News Detection using Natural Language Processing

---

## 🧠 Project Overview

Welcome to the **Fake News Detection System**! 🕵️‍♂️📰  
In an age of rampant misinformation, this project aims to classify news articles as **Fake 🛑** or **Real ✅** using the power of **Natural Language Processing (NLP)** and **Machine Learning (ML)**.

It employs a **Logistic Regression** model trained on the **Kaggle Fake and Real News Dataset**, achieving **~92% accuracy**!  
The solution was designed to be simple, lightweight, and practical – perfect for students and NLP enthusiasts.

---

## 📂 What's Inside?

This repository contains:

- 🧾 `my_fake_news_detection.ipynb` – Jupyter Notebook containing the complete implementation.
- 📊 `confusion_matrix.png` – Visualization of model performance using a confusion matrix.
- 📘 `explanation.md` – Brief summary of implementation (Phase 2 deliverable).
- 📄 `report.tex` – Full final report written in LaTeX (Phase 3 deliverable).

---

## ⚙️ How to Use This Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/HarshK0103/GENAi_Project_22BEC10034
cd GENAi_Project_22BEC10034
```

### 2️⃣ Set Up the Environment

Ensure you're using **Python 3.8+**.

Install the required libraries:

```bash
pip install pandas nltk scikit-learn numpy seaborn matplotlib
```

### 3️⃣ Download the Dataset

📥 The dataset is not included due to size limitations.

- Get it from [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Extract and place the files like this:

```
GENAi_Project_22BEC10034/
└── data/
    ├── Fake.csv
    └── True.csv
```

### 4️⃣ Run the Jupyter Notebook

Launch the notebook:

```bash
jupyter notebook my_fake_news_detection.ipynb
```

Then run all cells via **Cell > Run All**.

---

## 📊 Results

| Metric      | Fake News | Real News |
|-------------|-----------|-----------|
| Precision   | 0.91      | 0.93      |
| Recall      | 0.93      | 0.91      |
| F1-Score    | 0.92      | 0.92      |

🎯 **Overall Accuracy:** `92%`

📌 **Visualization:** Refer to `confusion_matrix.png`

🧪 **Sample Prediction:**

> `"Government claims new policy boosts economy, lacks evidence."`  
> → **Fake** 🛑

---

## 🔍 Key Features

- ✅ Real-time prediction of fake vs real news.
- 🧹 Clean preprocessing pipeline using NLTK.
- 🤖 Simple and interpretable ML model: Logistic Regression.
- 📈 Evaluation using Accuracy, Precision, Recall, and F1 Score.
- 🧪 Easily extendable to deep learning models like BERT, LSTM.

---

## 📦 Tools & Technologies Used

- 🐍 **Python 3.8+**
- 📚 **Libraries**: `pandas`, `nltk`, `scikit-learn`, `numpy`, `seaborn`, `matplotlib`
- 📊 **Dataset**: [Kaggle Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- 🧪 **Notebook**: Jupyter
- 🧠 **Optional**: Can be extended to include BERT (HuggingFace), Streamlit interface, or LSTM.

---

## 🧾 Notes

- This is a **simplified and efficient** version of more complex BERT/LSTM-based approaches.
- Designed to run on a **standard laptop** with minimal dependencies.
- Ideal for students or developers who want to get started with NLP-based classification.

---

## 🙏 Acknowledgments

- 📊 **Dataset:** Kaggle - Fake and Real News Dataset  
- 💡 **Inspiration:** Adapted from a downloaded project that utilized advanced models like BERT and LSTM. This implementation uses similar workflows but with improved simplicity and performance for academic use.

---

## ⭐️ Conclusion

Thanks for checking out the Fake News Detector!  
This project combines the power of NLP and ML to tackle misinformation and promote truth in media. 🌐🧠  
If you found this helpful, feel free to ⭐ the repository or share your ideas for improvements!

---

> _“In a time of deceit, telling the truth is a revolutionary act.” – George Orwell_
