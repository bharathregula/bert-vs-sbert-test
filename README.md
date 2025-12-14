# SBERT vs. BERT: Semantic Similarity Demonstration

This repository contains Python code demonstrating the superior performance of **Sentence-BERT (SBERT)** over vanilla **BERT** for the task of semantic textual similarity (STS). The demonstration uses the industry-standard **STS Benchmark (STS-B)** dataset.

## Why SBERT is Necessary

Vanilla BERT embeddings, while contextually rich, are not optimized for distance-based comparison (cosine similarity). This results in poor correlation with human judgment. SBERT uses a Siamese network and contrastive learning to explicitly train the model to cluster semantically similar sentences closely together in vector space.
