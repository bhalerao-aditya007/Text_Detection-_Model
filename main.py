from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import torch
from transformers import BertTokenizer, BertModel
import re
import numpy as np

app = FastAPI()

# Preprocess text, BERTEncoder, predict_text (same as above)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class BERTEncoder:
    def __init__(self, model_name="bert-base-uncased", max_length=128, batch_size=32):
        self.device = torch.device('cpu')
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def encode_batch(self, texts):
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
        if len(texts) == 0:
            raise ValueError("No texts to encode!")
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self.encode_batch(batch)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

def predict_text(text, model, encoder):
    processed = preprocess_text(text)
    if len(processed) == 0:
        return {"prediction": "UNKNOWN", "error": "Text too short"}
    embedding = encoder.encode_texts([processed])
    prediction = model.predict(embedding)[0]
    probability = model.predict_proba(embedding)[0]
    return {
        "prediction": "HARMFUL" if prediction == 1 else "SAFE",
        "harmful_probability": float(probability[1]),
        "safe_probability": float(probability[0])
    }

# Load model
try:
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    encoder = BERTEncoder()
except Exception as e:
    raise RuntimeError(f"Failed to load model or encoder: {str(e)}")

# FastAPI endpoint
class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input: TextInput):
    return predict_text(input.text, model, encoder)

@app.get("/")
async def root():
    return {"message": "Hate Speech Detector API. Use POST /predict with {'text': 'your text'}"}
