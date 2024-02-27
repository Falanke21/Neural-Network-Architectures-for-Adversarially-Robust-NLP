# How to design self-attention for a safer Transformer AI?

### Despite advancements in Deep Learning, adversarial attacks remain problematic, including NLP and transformer models. This project assesses various self-attention mechanisms to enhance transformer robustness against adversarial attacks in NLP.

## RNN & LSTM vs. Transformer:
Comparing sequential models like RNN and LSTM with the parallel processing of transformers.

## Adversarial Attacks:
Explanation of NLP-focused adversarial attacks and their impact.

## Experimental Setup:
We are utilizing Yelp-polarity sentiment analysis and TextAttack for robustness evaluation.

## Self-Attention Variants:
Brief overview of Additive Attention, Paas, Linformer, SimA, SOFT, CosFormer, and TransNormer.

## Ablation Studies:
Exploration of Diag attention and its impact on model robustness.

## Word Embeddings:
Comparison of custom, GloVe, and Counter-fitting word embeddings.

## Number of Heads:
Examining the influence of head number on transformer model robustness.

## ReVA & ReVCos:
Introduction of ReLU Value attention and ReLU Value CosFormer for enhanced robustness.

## Adversarial Training:
Results of adversarial training using Textfooler method.

## Scaling Capacity:
Discussion on the scalability of attention mechanisms in larger transformer models.
