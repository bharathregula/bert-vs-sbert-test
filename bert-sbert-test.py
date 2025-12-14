import numpy as np
import torch
from scipy.stats import spearmanr
from datasets import load_dataset
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the STS-B Dataset
# We load the test split as the gold standard for evaluation
print("Loading STS-B Dataset...")
dataset = load_dataset("stsb_multi_mt", "en", split="test")

# Extract the sentences and the ground truth scores (labels)
sentences1 = dataset["sentence1"]
sentences2 = dataset["sentence2"]
gold_scores = np.array(dataset["similarity_score"])

print(f"Dataset loaded with {len(gold_scores)} pairs.")
print("-" * 50)

# --- BERT Base Model Implementation ---

def get_bert_embeddings(sentences):
    """
    Generates sentence embeddings using Vanilla BERT with Mean Pooling.
    This is the most common naive approach for BERT sentence embeddings.
    """
    # Load the base BERT model and tokenizer
    # Using 'bert-base-uncased' as the base for the comparison
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize the sentences
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    
    # Run the model without calculating gradients (inference mode)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the token embeddings from the last layer
    token_embeddings = outputs.last_hidden_state  # shape: [batch_size, seq_len, 768]
    
    # Calculate Mean Pooling (the most common method)
    # Exclude padding tokens by using the attention mask
    input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # Mean Pooling: divide the summed embeddings by the number of actual tokens
    embeddings = sum_embeddings / sum_mask
    
    return embeddings.numpy()


# --- SBERT Model Implementation ---

def get_sbert_embeddings(sentences):
    """
    Generates semantically meaningful embeddings using a pre-trained SBERT model.
    We use a highly efficient variant like all-MiniLM-L6-v2 for this demo.
    """
    # Load a strong, pre-trained Sentence-Transformer model
    # This model was fine-tuned specifically for similarity tasks (on NLI + STS data)
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    # The .encode() method handles tokenization, forward pass, and pooling internally
    embeddings = sbert_model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


# --- Evaluation Function ---

def evaluate_model(embeddings1, embeddings2, gold_scores, model_name):
    """
    Calculates the cosine similarity between the generated embeddings
    and computes the Spearman's Rank Correlation with the ground truth scores.
    """
    # Compute Cosine Similarity between all pairs
    # Note: cosine_similarity returns a matrix; we take the diagonal for pair-wise scores
    sim_scores = cosine_similarity(embeddings1, embeddings2)
    print(sim_scores)
    predicted_scores = np.diag(sim_scores)
    print(predicted_scores)
    
    # Spearman's Rank Correlation is the standard metric for STS-B
    # It measures how well the ranking of similarity matches the human ranking
    spearman_corr, _ = spearmanr(gold_scores, predicted_scores)
    
    print(f"[{model_name} Result]")
    print(f"Spearman's Rank Correlation: {spearman_corr:.4f}")
    print("-" * 50)
    return spearman_corr

# --- Run Comparison ---

# Run 1: Traditional BERT (Mean Pooling)
print("Running Traditional BERT (Mean Pooling)...")
bert_embeddings1 = get_bert_embeddings(sentences1)
bert_embeddings2 = get_bert_embeddings(sentences2)
bert_corr = evaluate_model(bert_embeddings1, bert_embeddings2, gold_scores, "Traditional BERT")


# Run 2: Sentence-BERT (SBERT)
print("Running Sentence-BERT (SBERT)...")
sbert_embeddings1 = get_sbert_embeddings(sentences1)
sbert_embeddings2 = get_sbert_embeddings(sentences2)
sbert_corr = evaluate_model(sbert_embeddings1, sbert_embeddings2, gold_scores, "Sentence-BERT")


# --- 6. Final Summary ---
print("\n--- Summary of Results ---")
print(f"Traditional BERT (Mean Pooling) Correlation: {bert_corr:.4f}")
print(f"Sentence-BERT Correlation:                  {sbert_corr:.4f}")

if sbert_corr > bert_corr:
    print("\n SBERT demonstrates superior performance for semantic similarity.")
else:
    print("\n Unexpected result. SBERT is typically much better for this task.")
