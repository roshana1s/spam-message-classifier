# 🚫 Spam Message Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/roshana1s/spam-message-classifier)

A state-of-the-art spam message classifier built with **RoBERTa** transformer model, fine-tuned on multiple SMS spam datasets. This model achieves exceptional performance with **0.9963 F1-score** for spam detection and **99.88% overall accuracy**, making it ideal for real-world deployment in messaging platforms and content moderation systems.

## 🎯 Overview

This project develops an intelligent spam detection system using advanced natural language processing techniques. The classifier is designed to accurately distinguish between legitimate messages (ham) and spam content, with a focus on minimizing both false positives and false negatives.

### Key Features
- **🤖 Transformer-based Architecture**: Built on RoBERTa-base for superior text understanding
- **⚡ High Performance**: 0.9963 F1-score for spam detection, 99.88% overall accuracy
- **🔧 Hyperparameter Optimization**: Automated tuning using Optuna framework (25 trials)
- **⚖️ Class Imbalance Handling**: Weighted loss function for optimal training
- **📊 Comprehensive Evaluation**: Multiple metrics including precision, recall, and confusion matrix
- **🚀 Production-Ready**: Saved in Hugging Face format for easy deployment

## 📊 Model Performance

### Final Results on Test Set:
- **Overall Accuracy**: 99.88%
- **Weighted F1-Score**: 0.9988
- **Spam F1-Score**: 0.9963 ✅ (Exceeds 0.95 acceptance threshold)
- **Spam Precision**: 100.00% (Perfect precision - no false alarms)
- **Spam Recall**: 99.27% (High spam detection rate)
- **Ham Precision**: 99.86%
- **Ham Recall**: 100.00%

### Acceptance Criteria
> ✅ **Model Accepted**: The F1-score for spam class (0.9963) significantly exceeds our predefined acceptance threshold of 0.95, indicating exceptional performance for real-world deployment.

### Generalizability
> 📊 **Strong Generalization**: All performance metrics are evaluated on a **completely unseen test set** (15% of data, ~1,725 messages) that was never used during training or hyperparameter tuning, ensuring robust real-world performance and preventing overfitting.

## 🏗️ Architecture & Methodology

### Model Architecture
- **Base Model**: FacebookAI/roberta-base
- **Task**: Binary sequence classification (ham vs spam)
- **Fine-tuning**: Custom classification head with 2 output labels
- **Tokenization**: RoBERTa tokenizer with optimal sequence length

### Training Strategy
1. **Data Preprocessing**: SMS text cleaning and label encoding
2. **Tokenization**: Dynamic padding with optimal max length
3. **Class Balancing**: Weighted loss function to handle imbalanced dataset
4. **Hyperparameter Optimization**: Optuna-based automated tuning
5. **Evaluation**: Comprehensive metrics on held-out test set

### Hyperparameter Optimization
Used **Optuna** framework to optimize (25 trials):
- **Dropout rates**: Hidden dropout (0.1-0.3), Attention dropout (0.1-0.2)
- **Learning rate**: 1e-5 to 5e-5 range
- **Weight decay**: 0.0 to 0.1 regularization
- **Batch size**: 8, 16, or 32 samples
- **Gradient accumulation steps**: 1 to 4
- **Training epochs**: 2 to 5 epochs
- **Warmup ratio**: 0.05 to 0.1 for learning rate scheduling

**Best Parameters Found (Trial 17/25)**:
- Hidden dropout: 0.161
- Attention dropout: 0.116  
- Learning rate: 1.67e-05
- Weight decay: 0.0235
- Batch size: 16
- Gradient accumulation steps: 3
- Epochs: 4
- Warmup ratio: 0.0502

## 📁 Project Structure

```
spam-message-classifier/
├── data/
│   └── spam.csv
├── notebooks/
│   └── spam-message-classifier.ipynb    # Complete development notebook
├── README.md                           # This file
└── .gitignore                         # Git ignore rules
```

**Note**: The trained model and tokenizer are hosted on Hugging Face Hub at [roshana1s/spam-message-classifier](https://huggingface.co/roshana1s/spam-message-classifier)

## 🚀 Quick Start

### Installation

**Install required packages:**
```bash
pip install transformers torch
```

### Usage

**Load and use the trained model from Hugging Face:**
```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load the trained model and tokenizer from Hugging Face Hub
model = RobertaForSequenceClassification.from_pretrained("roshana1s/spam-message-classifier")
tokenizer = RobertaTokenizer.from_pretrained("roshana1s/spam-message-classifier")
```

## 📖 Dataset

**Sources**: 
1. **[SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)** from UCI Machine Learning Repository
2. **[SMS Phishing Dataset](https://data.mendeley.com/datasets/f45bkkt8pr/1)** by Sandhya Mishra & Devpriya Soni (2022)

**Combined Dataset Statistics**:
- **Total Messages**: 11,498 (entire combined dataset)
- **Ham Messages**: 9,669 (84.1%)
- **Spam Messages**: 1,829 (15.9%)
- **Average Message Length**: ~80 characters
- **Language**: English

**Dataset Split**:
- **Training Set**: 70% (~8,048 messages) - used for model training
- **Validation Set**: 15% (~1,725 messages) - used for hyperparameter tuning
- **Test Set**: 15% (~1,725 messages) - used for final evaluation (unseen data)

**Dataset Details**:
- **UCI SMS Spam**: Classic SMS spam detection dataset with ham/spam labels
- **Mendeley SMS Phishing**: Modern phishing detection dataset including smishing attacks
- **Combined Approach**: Merged datasets for comprehensive spam and phishing detection

**Preprocessing Steps**:
1. Label encoding (ham → 0, spam → 1, smishing → 1)
2. Text cleaning and normalization with Discord-specific preprocessing
3. Dataset merging and deduplication
4. Train/validation/test split (70/15/15)
5. Tokenization with RoBERTa tokenizer
6. Dynamic padding and truncation

## 🛠️ Technical Implementation

### Key Technologies
- **🤗 Transformers**: Hugging Face transformers library
- **🔥 PyTorch**: Deep learning framework
- **📊 Scikit-learn**: Evaluation metrics and preprocessing
- **🎯 Optuna**: Hyperparameter optimization
- **📈 Matplotlib/Seaborn**: Data visualization
- **🐼 Pandas**: Data manipulation

### Custom Features
- **Weighted Loss Function**: Handles class imbalance effectively
- **Label Smoothing**: 0.1 to prevent overconfidence
- **Custom Metrics**: Specialized spam detection metrics
- **Confusion Matrix Analysis**: Detailed error analysis
- **Class-specific Performance**: Separate metrics for ham and spam

## 📊 Detailed Results

### Confusion Matrix
```
          | Predicted Ham | Predicted Spam
----------|---------------|---------------
Actual Ham|     1451     |       0
Actual Spam|       2      |     272
```

### Performance Breakdown
- **True Positives (Spam correctly identified)**: 272
- **True Negatives (Ham correctly identified)**: 1451  
- **False Positives (Ham incorrectly flagged)**: 0
- **False Negatives (Spam missed)**: 2

### Error Analysis
- **False Positive Rate**: 0.00% (Perfect - no ham flagged as spam)
- **Miss Rate**: 0.73% (Extremely low spam miss rate)
- **False Alarm Rate**: 0.00% - ensuring no legitimate messages are incorrectly flagged

## 🎯 Use Cases

This spam classifier is ideal for:

### 💬 **Messaging Platforms**
- Discord bot moderation (Primary use case)
- SMS filtering systems
- Chat application content filtering

### 🛡️ **Content Moderation**
- Social media platforms
- Comment section filtering
- User-generated content screening

## 🔄 Model Deployment

### Hugging Face Hub
The trained model is available on Hugging Face Hub:
👉 **[roshana1s/spam-message-classifier](https://huggingface.co/roshana1s/spam-message-classifier)**

### Integration with Amy Discord Bot
This model serves as the core spam detection component for **Amy**, an intelligent Discord moderation bot that:
- Detects spam messages in real-time
- Provides automated content moderation
- Maintains server quality and user experience

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐
