import streamlit as st
import pickle
import torch
from transformers import BertTokenizer, BertModel
import re
import numpy as np

# Preprocess text function
def preprocess_text(text):
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# BERT Encoder class
class BERTEncoder:
    def __init__(self, model_name="bert-base-uncased", max_length=128, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.batch_size = batch_size
        st.write(f"Loading BERT model: {model_name} on {self.device}")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            st.error(f"Failed to load BERT model: {str(e)}")
            raise RuntimeError(f"Failed to load BERT model: {str(e)}")
    
    def encode_batch(self, texts):
        """Encode a batch of texts"""
        try:
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
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Error encoding batch: {str(e)}")
    
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
                'text': text,
                'prediction': 'UNKNOWN',
                'harmful_probability': 0.0,
                'safe_probability': 0.0,
                'error': 'Text too short after preprocessing'
            }
        
        embedding = encoder.encode_texts([processed])
        prediction = model.predict(embedding)[0]
        probability = model.predict_proba(embedding)[0]
        
        return {
            'text': text,
            'prediction': 'HARMFUL' if prediction == 1 else 'SAFE',
            'harmful_probability': float(probability[1]),
            'safe_probability': float(probability[0])
        }
    except Exception as e:
        return {
            'text': text,
            'prediction': 'ERROR',
            'harmful_probability': 0.0,
            'safe_probability': 0.0,
            'error': str(e)
        }

# Load the model
try:
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    encoder = BERTEncoder()
except Exception as e:
    st.error(f"Failed to load model or encoder: {str(e)}")
    st.stop()

# Streamlit UI
st.title("Hate Speech Detector")
st.write("Enter text to check if it's harmful or safe.")
text_input = st.text_area("Enter text to classify:", height=150)
if st.button("Predict"):
    if text_input:
        with st.spinner("Processing..."):
            result = predict_text(text_input, model, encoder)
            st.write(f"**Prediction**: {result['prediction']}")
            if 'error' not in result:
                st.write(f"**Harmful Probability**: {result['harmful_probability']:.4f}")
                st.write(f"**Safe Probability**: {result['safe_probability']:.4f}")
            else:
                st.error(f"Error: {result['error']}")
    else:
        st.warning("Please enter some text.")