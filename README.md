# PDF Question Answering with BGE & Matryoshka Fine-tuning

This project demonstrates a question-answering system built using Google's Gemini and Sentence Transformers. It utilizes the BGE (BAAI/bge-base-en-v1.5) model for embeddings and applies Matryoshka loss for fine-tuning, resulting in improved information retrieval capabilities. The system's performance is evaluated using the NDCG metric, which measures the quality of ranked retrieval results.

## Functionality

1. **PDF Processing:** Extracts text from PDF documents using `unstructured` library.
2. **Text Chunking:** Splits the extracted text into smaller chunks using `nltk` for processing by Gemini.
3. **Question & Answer Generation:** Employs Google's Gemini (`gemini-1.5-flash`) to generate question-answer pairs based on the text chunks.
4. **Dataset Creation:** Stores generated QA pairs in a Hugging Face Dataset format for efficient handling.
5. **Sentence Embeddings:** Utilizes Sentence Transformers to generate embeddings for questions and answers.
6. **Matryoshka Fine-tuning:** Fine-tunes the BGE model using Matryoshka loss to enhance retrieval accuracy, aiming to improve NDCG scores.
7. **Evaluation:** Employs `InformationRetrievalEvaluator` to assess the model's performance on question answering tasks, specifically using **NDCG@10** as the primary metric across various embedding dimensions. The NDCG metric measures the quality of ranked results, considering both relevance and position.



## Requirements

- Python 3.x
- Libraries: unstructured, nltk, datasets, sentence-transformers, google-generativeai, pandas, torch, pdfminer.six, scikit-learn


## Installation
bash !sudo apt-get install tesseract-ocr 
!pip install poppler-utils tesseract-ocr 
!pip install datasets sentence-transformers google-generativeai 
!pip install -q --user --upgrade pillow 
!pip install -q unstructured["all-docs"] pi_heif 
!pip install -q --upgrade unstructured 
!pip install --upgrade nltk 
!apt-get update 
!apt-get install poppler-utils -y 
!pip install pdfminer.six 
!apt-get install poppler-utils -y 

## Usage

1. Ensure your Google API key is set (see code comments).
2. Place PDF files in the `data` folder.
3. Run the notebook cells sequentially to process PDFs, generate QA pairs, fine-tune the model, and evaluate its performance using NDCG.


## Key Improvements

- Utilizes Matryoshka loss for fine-tuning, which leverages multiple embedding dimensions to improve retrieval and ultimately enhance NDCG scores.
- Employs a sequential evaluator to assess performance across different embedding sizes, providing a comprehensive view of NDCG across dimensions.
- Saves the fine-tuned model for later use, allowing for efficient deployment and further evaluation.


## Notes

- PDF processing assumes a certain format and quality. Corrupted PDFs may cause errors.
- Model performance and NDCG scores are dependent on the quality and relevance of the generated question-answer pairs.
- Higher NDCG scores indicate better retrieval performance, with a perfect score of 1.0 representing ideal ranking.
