import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a given PDF file.

    Args:
    - pdf_path (str): Path to the PDF file.

    Returns:
    - str: Extracted text from the PDF.
    """
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

# Function to extract relevant features from the text
def extract_features(text):
    """
    Extracts relevant features from the extracted text.

    Args:
    - text (str): Text extracted from the PDF.

    Returns:
    - dict: Extracted features including invoice number, date, and amount.
    """
    invoice_number = re.search(r"Invoice Number: (\d+)", text)
    date = re.search(r"Date: (\d{4}-\d{2}-\d{2})", text)
    amount = re.search(r"Amount: (\d+\.\d{2})", text)
    return {
        "invoice_number": invoice_number.group(1) if invoice_number else "",
        "date": date.group(1) if date else "",
        "amount": amount.group(1) if amount else ""
    }

# Function to compute cosine similarity between two text documents
def compute_cosine_similarity(text1, text2):
    """
    Computes cosine similarity between two text documents.

    Args:
    - text1 (str): First document text.
    - text2 (str): Second document text.

    Returns:
    - float: Cosine similarity score between the two texts.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()[0]

# Function to find the most similar invoice from the train set
def find_most_similar_invoice(test_text, train_texts):
    """
    Finds the most similar invoice in the train set compared to the test invoice.

    Args:
    - test_text (str): Text of the test invoice.
    - train_texts (list of str): List of texts from the train invoices.

    Returns:
    - tuple: (similarity score, index of the most similar invoice)
    """
    if not train_texts:
        return 0, -1
    similarities = [compute_cosine_similarity(test_text, train_text) for train_text in train_texts]
    return max(similarities), similarities.index(max(similarities))

# Main function to process invoices
def process_invoices(test_folder, train_folder):
    """
    Processes invoices in the test folder and finds their most similar counterparts in the train folder.

    Args:
    - test_folder (str): Path to the folder containing test invoices.
    - train_folder (str): Path to the folder containing train invoices.
    """
    # Load train invoices
    train_texts = []
    train_files = [f for f in os.listdir(train_folder) if f.endswith('.pdf')]
    for filename in train_files:
        path = os.path.join(train_folder, filename)
        text = extract_text_from_pdf(path)
        if text:
            train_texts.append(text)

    # Compare each test invoice with train invoices
    test_files = [f for f in os.listdir(test_folder) if f.endswith('.pdf')]
    for filename in test_files:
        test_path = os.path.join(test_folder, filename)
        test_text = extract_text_from_pdf(test_path)
        if test_text:
            similarity, best_match_index = find_most_similar_invoice(test_text, train_texts)
            best_match_file = train_files[best_match_index] if best_match_index >= 0 else "None"
            print(f"Test Invoice: {filename}")
            print(f"Most Similar Train Invoice: {best_match_file}")
            print(f"Similarity Score: {similarity:.4f}\n")

# Example usage
test_folder = "./documentsimilarity/test"
train_folder = "./documentsimilarity/train"
process_invoices(test_folder, train_folder)
