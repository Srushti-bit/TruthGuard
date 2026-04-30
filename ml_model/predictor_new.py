import torch
import torch.nn as nn
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import BertTokenizer, BertModel
import numpy as np
from datetime import datetime

# Setup NLTK
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# ============================================================================
# OBJECTIVE 3: BERT TEXT ENCODER (Deep Semantic Features)
# ============================================================================
class BertTextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(BertTextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, text, max_length=512):
        inputs = self.tokenizer(
            text, return_tensors='pt', max_length=max_length,
            truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.bert(**inputs)
        
        # Extract [CLS] token (768-dim)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.dropout(cls_token)

# ============================================================================
# OBJECTIVE 2: TIME-AWARE GCN (w = exp(-λΔt))
# ============================================================================
class TimeAwareGCN(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, output_dim=64, lambda_param=0.01):
        super(TimeAwareGCN, self).__init__()
        self.lambda_param = lambda_param # Adjusted to 0.01 for 500 shares
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def compute_edge_weights(self, timestamps):
        if not timestamps or len(timestamps) == 0: return None
        timestamps = np.array(timestamps)
        time_diffs = timestamps - timestamps[0] # Δt
        weights = np.exp(-self.lambda_param * time_diffs) # THE FORMULA
        return torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, node_features, edge_index, edge_weights=None):
        x = self.relu(self.fc1(node_features))
        x = self.dropout(x)
        x = self.fc2(x)
        if edge_weights is not None:
            weighted_x = x * edge_weights.unsqueeze(1)
            return weighted_x.mean(dim=0, keepdim=True)
        return x.mean(dim=0, keepdim=True)

# ============================================================================
# OBJECTIVE 4: HYBRID FRAMEWORK + MC DROPOUT
# ============================================================================
class HybridTruthGuard(nn.Module):
    def __init__(self, bert_encoder, gcn_model):
        super(HybridTruthGuard, self).__init__()
        self.bert_encoder = bert_encoder
        self.gcn_model = gcn_model
        # Late Fusion: 768 (BERT) + 64 (GCN) = 832
        self.fusion_layer = nn.Linear(768 + 64, 256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.4), # Active during MC Dropout
            nn.Linear(128, 2)
        )
    
    def forward(self, text, propagation_graph=None):
        bert_feat = self.bert_encoder(text)
        if propagation_graph:
            gcn_feat = self.gcn_model(propagation_graph['node_features'], 
                                    None, propagation_graph['edge_weights'])
        else:
            gcn_feat = torch.zeros(1, 64)
        
        combined = torch.cat([bert_feat, gcn_feat], dim=-1)
        combined = torch.relu(self.fusion_layer(combined))
        return self.classifier(combined)

    def monte_carlo_prediction(self, text, propagation_graph=None, num_passes=50):
        self.train() # Keep dropout active for MC sampling
        preds = []
        for _ in range(num_passes):
            logits = self.forward(text, propagation_graph)
            preds.append(torch.softmax(logits, dim=1).detach().numpy())
        
        self.eval()
        preds = np.array(preds)
        mean_pred = preds.mean(axis=0)[0]
        std_pred = preds.std(axis=0)[0]
        
        idx = np.argmax(mean_pred)
        return {'mean': mean_pred, 'conf': mean_pred[idx], 'unc': std_pred[idx], 'class': idx}

# ============================================================================
# STEP 5: INTERFACE & OBJECTIVE 1 (Early Detection Logic)
# ============================================================================
def build_propagation_graph(propagation_data, num_shares=500):
    # OBJECTIVE 1: Slice data to first N shares
    data = propagation_data[:num_shares]
    num_nodes = len(data)
    
    # Simple node features (Repeat reliability score to 64-dim)
    scores = np.array([p.get('reliability_score', 0.5) for p in data]).reshape(-1, 1)
    node_feat = torch.tensor(np.repeat(scores, 64, axis=1), dtype=torch.float32)
    
    # Calculate timestamps for Objective 2
    ts = [p.get('timestamp', datetime.now()) for p in data]
    ts_num = [(t - ts[0]).total_seconds() for t in ts]
    weights = TimeAwareGCN().compute_edge_weights(ts_num)
    
    return {'node_features': node_feat, 'edge_weights': weights, 'num_shares': num_nodes}

# Initialize global model
hybrid_model = HybridTruthGuard(BertTextEncoder(), TimeAwareGCN())

def predict_news(text, propagation_data=None, num_shares=500, monte_carlo_passes=50):
    # Preprocess
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    graph = None
    if propagation_data:
        graph = build_propagation_graph(propagation_data, num_shares)
    
    res = hybrid_model.monte_carlo_prediction(clean_text, graph, monte_carlo_passes)
    
    prediction = 'REAL' if res['class'] == 1 else 'FAKE'
    return {
        'prediction': prediction,
        'confidence': round(float(res['conf']) * 100, 2),
        'uncertainty': round(float(res['unc']) * 100, 2),
        'fake_prob': round(float(res['mean'][0]) * 100, 2),
        'real_prob': round(float(res['mean'][1]) * 100, 2),
        'num_shares_used': graph['num_shares'] if graph else 0,
        'method': 'hybrid_bert_gcn_mc_dropout'
    }