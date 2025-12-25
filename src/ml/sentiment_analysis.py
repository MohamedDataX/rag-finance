import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import pandas as pd
import numpy as np

class FinBERTAnalyzer:
    def __init__(self):
        print("Chargement du modèle FinBERT...")
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        self.model.eval()  # no training, que l'evaluation
        print("Modèle chargé.")

    def chunk_text(self, text, max_len=510):
        """
        Découpe le txt en morceaux de taille max
        BERT a une limite stricte de 512 tokens
        """
        tokens = self.tokenizer.tokenize(text)
        # On découpe en chunks de 510 pour laisser de la place aux tokens spéciaux ..
        chunks = [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)]
        return chunks

    def predict_sentiment(self, text):
        """
        Calcule le sentiment global d'un texte long
        Retourne un score entre -1 (-) & 1 (+)
        """
        if not text or len(text) < 50:
            return 0.0, {} 

        chunks = self.chunk_text(text)
        sentiments = []


        with torch.no_grad():
            for chunk in chunks:

                inputs = self.tokenizer.encode_plus(
                    chunk,
                    add_special_tokens=True,
                    return_tensors='pt',
                    is_split_into_words=True # important car on passe deja des tokens
                )

                outputs = self.model(**inputs)
                probs = softmax(outputs.logits, dim=1)
                
                score = probs[0].numpy()
                sentiments.append(score)

        if not sentiments:
            return 0.0, {}

        # moyenne des probabilités sur tous les chunks
        avg_scores = np.mean(sentiments, axis=0)
        
        # calcul : Positive - Negative
        # on ignore le neutre pour le signal directionnel, ou on le pondère)
        neutral, positive, negative = avg_scores[0], avg_scores[1], avg_scores[2]
        
        compound_score = positive - negative
        
        details = {
            "positive": float(positive),
            "negative": float(negative),
            "neutral": float(neutral),
            "chunks_processed": len(chunks)
        }
        
        return compound_score, details