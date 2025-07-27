# 🚫 Spam Message Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/roshana1s/spam-message-classifier)

A state-of-the-art spam message classifier built with **RoBERTa** transformer model, fine-tuned on the SMS Spam Collection Dataset. This model achieves exceptional performance with **97.30% F1-score** for spam detection and **99.28% overall accuracy**, making it ideal for real-world deployment in messaging platforms and content moderation systems.

## 🎯 Overview

This project develops an intelligent spam detection system using advanced natural language processing techniques. The classifier is designed to accurately distinguish between legitimate messages (ham) and spam content, with a focus on minimizing both false positives and false negatives.

### Key Features
- **🤖 Transformer-based Architecture**: Built on RoBERTa-base for superior text understanding
- **⚡ High Performance**: 97.30% F1-score for spam detection, 99.28% overall accuracy
- **🔧 Hyperparameter Optimization**: Automated tuning using Optuna framework  
- **⚖️ Class Imbalance Handling**: Weighted loss function for optimal training
- **📊 Comprehensive Evaluation**: Multiple metrics including precision, recall, and confusion matrix
- **🚀 Production-Ready**: Saved in Hugging Face format for easy deployment

## 📊 Model Performance

### Final Results on Test Set:
- **Overall Accuracy**: 99.28%
- **Weighted F1-Score**: 99.28%
- **Spam F1-Score**: 97.30% ✅ (Exceeds 95% acceptance threshold)
- **Spam Precision**: 98.18% (Low false positives)
- **Spam Recall**: 96.43% (High spam detection rate)
- **Ham Precision**: 99.45%
- **Ham Recall**: 99.72%

### Acceptance Criteria
> ✅ **Model Accepted**: The F1-score for spam class (97.30%) exceeds our predefined acceptance threshold of 95%, indicating reliable performance for real-world deployment.

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
Used **Optuna** framework to optimize:
- **Dropout rates**: Hidden dropout (0.2-0.4), Attention dropout (0.1-0.2)
- **Learning rate**: 1e-5 to 3e-5 range
- **Weight decay**: 0.01 to 0.1 regularization
- **Batch size**: 16 or 32 samples
- **Gradient accumulation steps**: 1 or 2
- **Training epochs**: 3 to 8 epochs
- **Warmup ratio**: 0.05 to 0.15 for learning rate scheduling

**Best Parameters Found**:
- Hidden dropout: 0.203
- Attention dropout: 0.118  
- Learning rate: 2.38e-05
- Weight decay: 0.0625
- Batch size: 16
- Gradient accumulation steps: 1
- Epochs: 8
- Warmup ratio: 0.108

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

**Source**: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from UCI Machine Learning Repository

**Statistics**:
- **Total Messages**: 5,572
- **Ham Messages**: 4,825
- **Spam Messages**: 747
- **Average Message Length**: ~80 characters
- **Language**: English

**Preprocessing Steps**:
1. Label encoding (ham → 0, spam → 1)
2. Text cleaning and normalization
3. Train/validation/test split (70/15/15)
4. Tokenization with RoBERTa tokenizer
5. Dynamic padding and truncation

## 🛠️ Technical Implementation

### Key Technologies
- **🤗 Transformers**: Hugging Face transformers library
- **🔥 PyTorch**: Deep learning framework
- **📊 Scikit-learn**: Evaluation metrics and preprocessing
- **🎯 Optuna**: Hyperparameter optimization
- **📈 Matplotlib/Seaborn**: Data visualization
- **🐼 Pandas**: Data manipulation

### Training Configuration
- **GPU**: NVIDIA Tesla T4 (if available)
- **Mixed Precision**: FP16 for memory efficiency
- **Evaluation Strategy**: Step-based evaluation every 200 steps
- **Best Model Selection**: Based on spam F1-score
- **Early Stopping**: Load best model at end

### Custom Features
- **Weighted Loss Function**: Handles class imbalance effectively
- **Custom Metrics**: Specialized spam detection metrics
- **Confusion Matrix Analysis**: Detailed error analysis
- **Class-specific Performance**: Separate metrics for ham and spam

## 📊 Detailed Results

### Confusion Matrix
```
          | Predicted Ham | Predicted Spam
----------|---------------|---------------
Actual Ham|      722     |       2
Actual Spam|       4      |     108
```

### Performance Breakdown
- **True Positives (Spam correctly identified)**: 108
- **True Negatives (Ham correctly identified)**: 722  
- **False Positives (Ham incorrectly flagged)**: 2
- **False Negatives (Spam missed)**: 4

### Error Analysis
- **False Positive Rate**: 0.28% (Ham incorrectly flagged as spam)
- **Miss Rate**: 3.57% (Spam messages missed)
- **False Alarm Rate**: Very low, ensuring minimal legitimate messages are flagged

## 🎯 Use Cases

This spam classifier is ideal for:

### 💬 **Messaging Platforms**
- Discord bot moderation (Primary use case)
- SMS filtering systems
- Chat application content filtering

### 📧 **Email Security**
- Email spam detection
- Corporate message filtering
- Customer service automation

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


##  Acknowledgments

- **UCI Machine Learning Repository** for the SMS Spam Collection Dataset
- **Hugging Face** for the transformers library and model hosting
- **Facebook AI** for the RoBERTa base model
- **Optuna** team for the hyperparameter optimization framework

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐