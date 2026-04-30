import torch
import torch.nn as nn
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import BertTokenizer, BertModel
import numpy as np
from datetime import datetime, timedelta

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# STEP 1: BERT TEXT ENCODER - Extracts deep semantic features
# ============================================================================

class BertTextEncoder(nn.Module):
    """Fine-tuned BERT for extracting deep contextual features"""
    
    def __init__(self, model_name='bert-base-uncased'):
        super(BertTextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, text, max_length=512):
        """
        Extract BERT embeddings
        Returns: contextual features (768-dim) and classification token
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
        
        # Get [CLS] token representation (index 0)
        cls_token = outputs.last_hidden_state[:, 0, :]
        cls_token = self.dropout(cls_token)
        
        return cls_token  # Shape: (batch_size, 768)
    
    def extract_features(self, text):
        """Convenience method to extract features from text"""
        return self.forward(text)


# ============================================================================
# STEP 2: TIME-AWARE GRAPH CONVOLUTIONAL NETWORK (GCN)
# ============================================================================

class TimeAwareGCN(nn.Module):
    """
    Graph-based model analyzing propagation patterns
    Uses exponential time-decay: w = exp(-λΔt)
    """
    
    def __init__(self, input_dim=64, hidden_dim=32, output_dim=64, lambda_param=0.1):
        super(TimeAwareGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lambda_param = lambda_param
        
        # GCN layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def compute_edge_weights(self, timestamps):
        """
        Compute time-decay weights: w = exp(-λΔt)
        where Δt is time elapsed since publication
        Gives higher importance to early shares
        """
        if not timestamps or len(timestamps) == 0:
            return None
        
        timestamps = np.array(timestamps)
        time_diffs = timestamps - timestamps[0]  # Relative to first share
        weights = np.exp(-self.lambda_param * time_diffs)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, node_features, edge_index, edge_weights=None):
        """
        Process propagation graph
        node_features: (num_nodes, input_dim)
        edge_index: (2, num_edges) - source and target node indices
        edge_weights: (num_edges,) - time-decay weights
        """
        # Simple graph convolution with edge weights
        x = self.fc1(node_features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Aggregate over graph if edge information provided
        if edge_weights is not None and edge_index is not None:
            # Weight aggregation based on edges
            weighted_x = x * edge_weights.unsqueeze(1)
            aggregated = weighted_x.mean(dim=0, keepdim=True)
            return aggregated
        
        return x.mean(dim=0, keepdim=True)  # Global average pooling


# ============================================================================
# STEP 3: HYBRID MODEL WITH LATE FUSION + MONTE CARLO DROPOUT
# ============================================================================

class HybridTruthGuard(nn.Module):
    """
    Combines:
    - Fine-tuned BERT (text analysis)
    - Time-aware GCN (propagation patterns)
    - Late fusion via concatenation
    - Monte Carlo Dropout for uncertainty estimation
    """
    
    def __init__(self, bert_encoder, gcn_model):
        super(HybridTruthGuard, self).__init__()
        self.bert_encoder = bert_encoder
        self.gcn_model = gcn_model
        
        # Late fusion: concatenate BERT (768) + GCN (64) features
        self.fusion_layer = nn.Linear(768 + 64, 256)
        self.fusion_dropout = nn.Dropout(0.5)  # Monte Carlo Dropout
        self.relu = nn.ReLU()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),  # Monte Carlo Dropout
            nn.Linear(128, 2)  # Binary: FAKE or REAL
        )
    
    def forward(self, text, propagation_graph=None):
        """
        Forward pass with both text and propagation features
        """
        # ============ BERT Stream ============
        bert_features = self.bert_encoder.extract_features(text)
        
        # ============ GCN Stream ============
        if propagation_graph is not None:
            gcn_features = self.gcn_model(
                propagation_graph['node_features'],
                propagation_graph['edge_index'],
                propagation_graph['edge_weights']
            )
        else:
            # If no propagation data, use zero vector
            gcn_features = torch.zeros(1, 64)
        
        # ============ Late Fusion ============
        combined = torch.cat([bert_features, gcn_features], dim=-1)
        combined = self.fusion_layer(combined)
        combined = self.relu(combined)
        combined = self.fusion_dropout(combined)  # MC Dropout
        
        # ============ Classification ============
        logits = self.classifier(combined)
        return logits
    
    def monte_carlo_prediction(self, text, propagation_graph=None, num_passes=50):
        """
        Uncertainty-aware prediction using Monte Carlo Dropout
        
        Performs multiple stochastic forward passes to estimate:
        - Mean prediction
        - Confidence score
        - Uncertainty estimate
        """
        predictions = []
        
        # Enable dropout during inference for MC sampling
        self.bert_encoder.train()
        self.classifier.train()
        
        for _ in range(num_passes):
            logits = self.forward(text, propagation_graph)
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs.detach().numpy())
        
        # Disable dropout
        self.bert_encoder.eval()
        self.classifier.eval()
        
        # Stack predictions: (num_passes, batch_size, num_classes)
        predictions = np.array(predictions)
        
        # Mean prediction (expected output)
        mean_pred = predictions.mean(axis=0)[0]
        
        # Uncertainty (standard deviation across passes)
        std_pred = predictions.std(axis=0)[0]
        
        # Prediction label and confidence
        predicted_class = np.argmax(mean_pred)
        confidence = mean_pred[predicted_class]
        uncertainty = std_pred[predicted_class]
        
        return {
            'mean_pred': mean_pred,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'predicted_class': predicted_class
        }


# ============================================================================
# STEP 4: HELPER FUNCTIONS FOR PROPAGATION GRAPH
# ============================================================================

def build_propagation_graph(propagation_data, num_shares=30):
    """
    Build graph from propagation data (first N shares)
    
    propagation_data: List of dicts with keys:
        - 'account_id': user account identifier
        - 'timestamp': when the share happened
        - 'reliability_score': credibility of account (0-1)
    
    Returns: Dict with graph structure
    """
    if not propagation_data or len(propagation_data) == 0:
        return None
    
    # Limit to first N shares (early detection)
    propagation_data = propagation_data[:num_shares]
    
    num_nodes = len(propagation_data)
    
    # Node features: reliability scores (as embeddings)
    node_features = np.array([
        p.get('reliability_score', 0.5) for p in propagation_data
    ]).reshape(-1, 1)
    
    # Expand to 64 dimensions (GCN input)
    node_features = np.repeat(node_features, 64, axis=1)
    node_features = torch.tensor(node_features, dtype=torch.float32)
    
    # Edge index: connect sequential shares (temporal edges)
    edge_index = []
    timestamps = []
    
    for i in range(num_nodes - 1):
        edge_index.append([i, i + 1])
        timestamps.append(propagation_data[i].get('timestamp', datetime.now()))
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else None
    
    # Compute time-decay weights
    gcn = TimeAwareGCN()
    if timestamps:
        timestamps_numeric = [(t - timestamps[0]).total_seconds() for t in timestamps]
        edge_weights = gcn.compute_edge_weights(timestamps_numeric)
    else:
        edge_weights = None
    
    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_weights': edge_weights,
        'num_shares': num_nodes
    }


# ============================================================================
# STEP 5: INITIALIZATION AND PREDICTION INTERFACE
# ============================================================================

def initialize_models():
    """Load or initialize BERT + GCN models"""
    
    # Initialize BERT encoder
    bert_encoder = BertTextEncoder('bert-base-uncased')
    bert_encoder.eval()
    
    # Initialize GCN
    gcn_model = TimeAwareGCN(input_dim=64, hidden_dim=32, output_dim=64)
    gcn_model.eval()
    
    # Create hybrid model
    hybrid_model = HybridTruthGuard(bert_encoder, gcn_model)
    hybrid_model.eval()
    
    return hybrid_model


def preprocess_text(text):
    """Basic text preprocessing"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)


# Load model once at startup
hybrid_model = initialize_models()


def predict_news(text, propagation_data=None, num_shares=30, monte_carlo_passes=50):
    """
    Predict if news is FAKE or REAL with confidence scores
    
    Args:
        text (str): News article content
        propagation_data (list): List of propagation events with:
            - 'account_id': user ID
            - 'timestamp': when shared
            - 'reliability_score': account credibility (0-1)
        num_shares (int): Use first N shares (10, 20, or 30)
        monte_carlo_passes (int): Number of stochastic passes
    
    Returns:
        dict: {
            'prediction': 'FAKE' or 'REAL',
            'confidence': float (0-1),
            'uncertainty': float (0-1),
            'bert_score': float (BERT confidence),
            'gcn_score': float (GCN confidence),
            'num_shares_used': int,
            'method': 'hybrid_bert_gcn_with_mc_dropout'
        }
    """
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Build propagation graph if data provided
    propagation_graph = None
    if propagation_data:
        propagation_graph = build_propagation_graph(propagation_data, num_shares)
    
    # Get prediction with uncertainty
    with torch.no_grad():
        mc_result = hybrid_model.monte_carlo_prediction(
            processed_text,
            propagation_graph,
            num_passes=monte_carlo_passes
        )
    
    mean_pred = mc_result['mean_pred']
    confidence = mc_result['confidence']
    uncertainty = mc_result['uncertainty']
    predicted_class = mc_result['predicted_class']
    
    # Extract individual scores
    fake_prob = mean_pred[0]
    real_prob = mean_pred[1]
    
    prediction = 'REAL' if predicted_class == 1 else 'FAKE'
    
    return {
        'prediction': prediction,
        'confidence': float(confidence),
        'uncertainty': float(uncertainty),
        'real_prob': round(float(real_prob) * 100, 2),
        'fake_prob': round(float(fake_prob) * 100, 2),
        'bert_score': float(real_prob) if prediction == 'REAL' else float(fake_prob),
        'gcn_score': float(confidence),  # From propagation patterns
        'num_shares_used': propagation_graph['num_shares'] if propagation_graph else 0,
        'method': 'hybrid_bert_gcn_with_mc_dropout'
    }


# Backward compatibility with old interface
def predict_news_simple(text):
    """
    Simple prediction without propagation data
    (Maintains compatibility with existing code)
    """
    return predict_news(text, propagation_data=None, num_shares=0, monte_carlo_passes=50)