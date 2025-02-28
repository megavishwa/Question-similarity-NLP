# Patient FAQ Assistant for Postpartum Depression

![Medical FAQ System](https://img.shields.io/badge/NLP-Medical%20FAQ%20System-blue)
![Python](https://img.shields.io/badge/Python-3.9-green)
![SpaCy](https://img.shields.io/badge/SpaCy-3.5.2-brightgreen)
![BERT](https://img.shields.io/badge/BERT-Sentence%20Similarity-orange)

## 📌 Problem Statement

After a clinical visit, patients often struggle to understand medications, doctor notes, physiotherapy instructions, and next visit summaries. The provided After Visit Summary (AVS) can be overwhelming and difficult to navigate.

To address this issue, we designed a FAQ-based patient assistant UX tailored for individuals suffering from Postpartum Depression (PPD). This project utilizes Natural Language Processing (NLP) techniques to automatically detect question similarity, making it easier for patients to find relevant information quickly.

## 🔍 What is Question Similarity?

Question similarity detection is a key NLP task that involves several steps:
- Data Collection (Open-source data from UNC Health)
- Exploratory Data Analysis
- Pre-processing & Feature Extraction
- Model Training & Evaluation

By leveraging pre-processing techniques such as tokenization, stopword removal, lemmatization, and named entity recognition, we improve the accuracy of similarity detection.

### 🔹 Representation Techniques:

To compare questions effectively, we explored:
- TF-IDF, Bag-of-Words, and N-grams
- Word Embeddings (Word2Vec, GloVe)
- Contextualized Word Embeddings (BERT, ELMo, Sentence-BERT)
- Universal Sentence Encoder for Transfer Learning

## ⚙️ Methodology

We trained multiple models to classify question similarity:
- ✅ BERT Model – To generate sentence embeddings
- ✅ Linear SVM, Random Forest, Logistic Regression – For classification
- ✅ Sentence Length Analysis – Additional feature extraction

### 📊 Model Evaluation

- All models performed similarly within the same time frame
- Linear SVM achieved 73% accuracy
- Random Forest achieved 64% accuracy
- Logistic Regression achieved 73% accuracy
- Parameter fine-tuning is required for better performance

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- Google Colab (recommended for GPU support)

### Required Libraries
```bash
pip install -U spacy
pip install openpyxl
pip install wordcloud
pip install matplotlib
pip install numpy
pip install pandas
pip install torch
pip install scikit-learn
pip install sentence-transformers
python -m spacy download en_core_web_sm
```

### How to Run the Script

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/patient-faq-assistant.git
   cd patient-faq-assistant
   ```

2. **Upload the dataset**
   - Place your Medical Question Pairs dataset in Excel format in the appropriate directory
   - The code expects a dataset with columns for question pairs and similarity labels

3. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook Patient_FAQ_Assistant.ipynb
   ```
   
4. **Using Google Colab (Alternative)**
   - Upload the notebook to Google Colab
   - Mount your Google Drive: `from google.colab import drive; drive.mount('/content/drive')`
   - Set the path to your dataset file
   - Run all cells

### Code Structure

The notebook contains the following key sections:
1. **Imports and Setup**: Loading necessary libraries and SpaCy models
2. **Data Loading**: Reading and structuring the dataset
3. **Preprocessing**: Cleaning text data and handling punctuation
4. **Feature Extraction**: Generating word embeddings
5. **Model Training**: Training and evaluating different classifiers
6. **Visualization**: Creating wordclouds and analyzing the dataset

## 📊 Results & Analysis

The project used SpaCy and Sentence-BERT embeddings to represent questions, followed by classification models to determine similarity:

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Linear SVM | 73.0% | 0.70 | 0.77 | 0.73 | 5683s |
| Random Forest | 64.4% | 0.62 | 0.66 | 0.64 | 6350s |
| Logistic Regression | 73.1% | 0.70 | 0.77 | 0.73 | 6649s |

The models demonstrated good performance, particularly with medical terminology and contextual understanding.

## 📌 Conclusion

The project demonstrated the effectiveness of BERT embeddings for question similarity detection in a healthcare setting. However, further optimization and hyperparameter tuning can improve the accuracy of predictions.

## 🚀 Future Improvements

- Fine-tune BERT for better performance
- Implement real-time patient queries via chatbot integration
- Expand dataset for better generalization
- Develop domain-specific embeddings for medical terminology
- Add web interface for easier interaction

## 📂 Project Files

- 📜 `Patient_FAQ_Assistant.ipynb` – Complete implementation
- 📊 Wordcloud visualizations for data analysis
- 📄 Dataset – Processed patient FAQ data (not included due to privacy)

## 👥 Team

- Mega Viswanathan

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

🌟 If you find this project helpful, give it a ⭐ on GitHub!

