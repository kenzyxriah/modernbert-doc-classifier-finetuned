# ModernBERT Document Compliance Classifier

A modern, high-performance document compliance classification system built on top of the **ModernBERT** architecture. This repository handles end-to-end PDF/document ingestion, token-optimized text extraction, caching, and model inference (both zero-shot fallback and supervised fine-tuned classification) to determine document compliance levels.


## 📁 Repository Structure

```text
├── lib/
│   ├── ml/
│   │   ├── document_classifier/
│   │   │   ├── classifier.py      # Classification wrapper & zero-shot logic
│   │   │   └── train.py           # PyTorch/Trainer setup for ModernBERT fine-tuning
│   │   └── pipeline.py            # Cached Hugging Face Transformers pipeline loader
│   ├── chat.py                    # Supplementary chat logic
│   ├── file_utils.py              # File downloader, caching, and tiktoken-based capping
│   └── parser.py                  # Docling parser for converting PDFs to Markdown
├── train_test/
│   ├── classifier_train.csv       # Training dataset
│   ├── classifier_test.csv        # Evaluation dataset
│   ├── split_data.py              # Splits raw data into train/test subsets
│   ├── generate_massive_data.py   # Synthetic data generation tool
│   └── colab_train.ipynb          # Jupyter notebook for notebook training/experiments
├── main.py                        # FastAPI application entry points and routes
├── auth.py                        # API Key authorization mechanism
├── utils.py                       # Logging and Redis connection configurations
├── run.py                         # Startup script for Uvicorn
└── requirements.txt               # Main dependencies
```

---

## 🏷️ Compliance Categories

Documents are classified into four major security and compliance tiers, defined as follows:

| Category | Description | Confidence Score |
| :--- | :--- | :--- |
| **Public** | Publicly accessible, safe for external marketing, and open to anyone. | Dynamic (Supervised / Zero-shot) |
| **Internal** | Company announcements, team building, and safe employee information. | Dynamic (Supervised / Zero-shot) |
| **Confidential** | Confidential business data, financial reports, and executive-level secrets. | Dynamic (Supervised / Zero-shot) |
| **Restricted** | Highly restricted IT architecture, root credentials, and severe security risks. | Dynamic (Supervised / Zero-shot) |

---

## ⚙️ Installation & Setup

### 1. Prerequisites
*   Python 3.10+
*   Redis Server (running locally on port `6379`)

### 2. Install Dependencies
Install all package dependencies via `pip`:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory and add the following keys:
```env
HF_TOKEN=your_huggingface_write_token
X_API_KEY=your_secure_api_key_for_fastapi
```

---

## 🖥️ Running the API Server

Start the FastAPI application by running:
```bash
python run.py
```
The server will start at `http://127.0.0.1:8010`. Interactive Swagger docs will be hosted at `http://127.0.0.1:8010/docs`.

### Protected API Endpoints

All classification endpoints require the `X-API-KEY` header matching the key configured in your `.env`.

#### 1. Classify a Document (`POST /api/v1/BERT/classify`)
Downloads the document from the provided URL, parses it, caps it using `tiktoken`, and classifies it.

*   **Request Header**: `X-API-KEY: <your_key>`
*   **Request Payload**:
    ```json
    {
      "document": "https://example.com/path/to/document.pdf"
    }
    ```
*   **Success Response**:
    ```json
    {
      "status": "success",
      "data": {
        "category": "Confidential",
        "confidence": 0.897
      },
      "message": "Compliance category retrieved"
    }
    ```

#### 2. Manual Classification Override (`PATCH /api/v1/BERT/update-doc-category`)
Forces an override for a document's classification in the cache.

*   **Request Header**: `X-API-KEY: <your_key>`
*   **Request Payload**:
    ```json
    {
      "document": "https://example.com/path/to/document.pdf",
      "category": "Restricted"
    }
    ```
*   **Success Response**:
    ```json
    {
      "status": "success",
      "data": {
        "category": "Restricted",
        "confidence": 1.0,
        "updated": true
      },
      "message": "Compliance category updated"
    }
    ```

---

## 🧠 Fine-Tuning the Model

The codebase includes a training script to fine-tune the ModernBERT base model on custom compliance datasets.

### Dataset Requirements
The training function expects a `datasets.Dataset` object with:
*   `text`: The parsed text of the document.
*   `label`: The integer ID of the target category.

### Triggering Training
You can run a fine-tuning job programmatically or via notebook:
```python
from lib.ml.document_classifier.train import train_model
from datasets import Dataset

# Load your training dataset (e.g., from CSV)
# dataset = Dataset.from_csv("train_test/classifier_train.csv")

# Define target labels
id2label = {
    0: "Public",
    1: "Internal",
    2: "Confidential",
    3: "Restricted"
}

# Run training and optionally push weights to Hugging Face Hub
train_model(
    dataset=dataset,
    id2label=id2label,
    epochs=3,
    batch_size=8,
    push_to_hub=True
)
```
