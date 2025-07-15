
# 📰 Fake News Detection using Python

This project aims to detect **Fake vs Real News** using **Logistic Regression** and **TF-IDF Vectorization**. The dataset is preprocessed, analyzed visually, and then used to train a supervised ML model. Evaluation metrics and graphs are included.

---

## 📁 Project Structure

```
fake_news_detection/
├── main.py                  # Main Python script
├── requirements.txt         # Required libraries
├── news_dataset.csv         # Dataset (uploaded by user)
├── README.md                # Project documentation
├── model/
│   └── logistic_model.pkl   # Saved ML model
└── visuals/
    └── data_distribution.png  # Graph of label distribution
```

---

## 🚀 How to Run This Project

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Run the main script:**

```bash
python main.py
```

3. **Outputs:**
   - Trained Logistic Regression model
   - Accuracy and classification metrics
   - Graph saved to `visuals/data_distribution.png`

---

## 📊 Graph Output Example

This bar graph shows how many fake vs real news entries are present in the dataset:

![Data Distribution](visuals/data_distribution.png)

---

## 📌 Tools & Libraries Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- NLTK
- TF-IDF Vectorization
- Logistic Regression

---

## ✅ Features

- Data cleaning and preprocessing
- Visualization of dataset distribution
- TF-IDF for text feature extraction
- Logistic Regression model for classification
- Model saving using `pickle`

---

## 📈 Model Performance

After training, the model provides:

- **Classification report**
- **Accuracy score**
- **Confusion matrix**

---

## 🙋‍♂️ Created By

**Umesh Rajput**  
BTech | Artificial Intelligence & Machine Learning  
St. Andrew's Institute of Technology and Management

---

## 📎 License

This project is for educational purposes only.
