# Patient FAQ Assistant for Postpartum Depression  

## 📌 Problem Statement  
After a clinical visit, patients often struggle to understand medications, doctor notes, physiotherapy instructions, and next visit summaries. The provided After Visit Summary (AVS) can be overwhelming and difficult to navigate.  

To address this issue, we designed a **FAQ-based patient assistant UX** tailored for individuals suffering from **Postpartum Depression (PPD)**. This project utilizes **Natural Language Processing (NLP)** techniques to **automatically detect question similarity**, making it easier for patients to find relevant information quickly.  

## 🔍 What is Question Similarity?  
Question similarity detection is a key **NLP task** that involves several steps:  

- **Data Collection** (Open-source data from UNC Health)  
- **Exploratory Data Analysis**  
- **Pre-processing & Feature Extraction**  
- **Model Training & Evaluation**  

By leveraging **pre-processing techniques** such as tokenization, stopword removal, lemmatization, and named entity recognition, we improve the accuracy of similarity detection.  

### 🔹 Representation Techniques:  
To compare questions effectively, we explored:  
- **TF-IDF, Bag-of-Words, and N-grams**  
- **Word Embeddings (Word2Vec, GloVe)**  
- **Contextualized Word Embeddings (BERT, ELMo, Sentence-BERT)**  
- **Universal Sentence Encoder for Transfer Learning**  

## ⚙️ Methodology  
We trained multiple models to classify question similarity:  
✅ **BERT Model** – To generate sentence embeddings  
✅ **Linear SVM, Random Forest, Logistic Regression** – For classification  
✅ **Sentence Length Analysis** – Additional feature extraction  

### 📊 Model Evaluation  
- All models performed **similarly within the same time frame**  
- **Parameter fine-tuning** is required for better performance  

## 📌 Conclusion  
The project demonstrated the effectiveness of **BERT embeddings** for question similarity detection in a healthcare setting. However, further optimization and **hyperparameter tuning** can improve the accuracy of predictions.  

## 🚀 Future Improvements  
- Fine-tune **BERT** for better performance  
- Implement **real-time** patient queries via chatbot integration  
- Expand dataset for **better generalization**  

## 📂 Project Files  
- 📜 **Jupyter Notebook** – Model implementation  
- 📊 **Project Slides** – Detailed explanation  
- 📄 **Dataset** – Processed patient FAQ data  

## 👥 Team  
- **Mega Viswanathan**  

---  

🌟 **If you find this project helpful, give it a ⭐ on GitHub!**  

