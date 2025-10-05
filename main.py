from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import torch
from transformers import BertTokenizer, BertModel
import re
import numpy as np

app = FastAPI()

# Preprocess text function
def preprocess_text(text):
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# BERT Encoder class
class BERTEncoder:
    def __init__(self, model_name="bert-base-uncased", max_length=128, batch_size=32):
        self.device = torch.device('cpu')  # Heroku free tier is CPU-only
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def encode_batch(self, texts):
        """Encode a batch of texts"""
        encoded = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings
    
    def encode_texts(self, texts):
        """Encode all texts in batches"""
        if len(texts) == 0:
            raise ValueError("No texts to encode!")
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self.encode_batch(batch)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

# Inference function
def predict_text(text, model, encoder):
    """Predict if a text is harmful or safe"""
    try:
        processed = preprocess_text(text)
        if len(processed) == 0:
            return {
                "prediction": "UNKNOWN",
                "error": "Text too short after preprocessing"
            }
        embedding = encoder.encode_texts([processed])
        prediction = model.predict(embedding)[0]
        probability = model.predict_proba(embedding)[0]
        return {
            "prediction": "HARMFUL" if prediction == 1 else "SAFE",
            "harmful_probability": float(probability[1]),
            "safe_probability": float(probability[0])
        }
    except Exception as e:
        return {
            "prediction": "ERROR",
            "harmful_probability": 0.0,
            "safe_probability": 0.0,
            "error": str(e)
        }

# Load model (updated to root directory)
try:
    with open('xgboost_model.pkl', 'rb') as f:  # Changed from 'models/xgboost_model.pkl'
        model = pickle.load(f)
    encoder = BERTEncoder()
except Exception as e:
    raise RuntimeError(f"Failed to load model or encoder: {str(e)}")

# FastAPI endpoints
class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input: TextInput):
    return predict_text(input.text, model, encoder)
