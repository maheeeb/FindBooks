# FindBooks

_A semantic book recommendation engine that suggests similar books based on description embeddings, powered by BERT and Streamlit._

## 🔍 Overview
This project recommends books by comparing:
- **BERT embeddings** of book descriptions (pre-computed)

Deployed as a web app using Streamlit for easy access.

## 🛠️ Features
- **Semantic search**: Finds books with similar themes, not just keyword matches
- **Pre-computed embeddings**: Fast loading with PyTorch `.pt` files
- **Hybrid filtering**: Combine BERT with metadata (genre/ratings) [SOON]
- **User feedback**: Rate recommendations to improve future results [SOON]

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation
```bash
git clone https://github.com/your-username/book-recommender.git
cd book-recommender
pip install -r requirements.txt
