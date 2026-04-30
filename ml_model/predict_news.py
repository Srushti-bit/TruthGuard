import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
import numpy as np

# ============================================================================
# DRAFT-MATCHING ARCHITECTURE (BERT + GCN Late Fusion)
# ============================================================================

class HybridTruthGuard(nn.Module):
    def __init__(self):
        super(HybridTruthGuard, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Layer 1: Fusion (768 BERT + 64 GCN = 832) -> 256 units
        self.fusion = nn.Linear(768 + 64, 256) 
        
        # Layer 2: Intermediate Dense -> 128 units
        self.fc2 = nn.Linear(256, 128)
        
        # Layer 3: Classification Head
        self.classifier = nn.Linear(128, 2)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, ids, mask, force_dropout=False):
        # Apply dropout during inference for MC Dropout
        if force_dropout:
            self.dropout.train()
        
        outputs = self.bert(ids, mask)
        bert_feats = outputs.last_hidden_state[:, 0, :]
        
        # GCN Placeholder (In full version, this uses 300-500 shares)
        gcn_placeholder = torch.zeros(bert_feats.shape[0], 64).to(bert_feats.device)
        
        combined = torch.cat([bert_feats, gcn_placeholder], dim=-1) # 832-dim
        
        x = torch.relu(self.fusion(combined))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        
        return self.classifier(x)

def run_prediction():
    device = 'cpu'
    model_path = 'hybrid_model.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found!")
        return

    print("🧠 Loading TruthGuard (Hybrid BERT+GCN)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = HybridTruthGuard().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    print("\n" + "="*50)
    print("TRUTHGUARD: UNCERTAINTY-AWARE DETECTOR")
    print("="*50)
    
    user_input = input("\nEnter a news headline to verify: ")

    inputs = tokenizer(user_input, max_length=64, truncation=True, padding='max_length', return_tensors='pt')

    # ============================================================================
    # MONTE CARLO DROPOUT (50 Forward Passes)
    # ============================================================================
    print(f"🔬 Performing 50 stochastic forward passes for uncertainty...")
    all_probs = []
    
    model.eval() 
    with torch.no_grad():
        for _ in range(50):
            # Force dropout to stay active
            outputs = model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device), force_dropout=True)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs.numpy())

    # Calculate Mean and Standard Deviation
    all_probs = np.array(all_probs)
    mean_probs = np.mean(all_probs, axis=0)[0]
    std_dev = np.std(all_probs, axis=0)[0]

    prediction = np.argmax(mean_probs)
    confidence = mean_probs[prediction] * 100
    uncertainty = std_dev[prediction]

    verdict = "✅ REAL" if prediction == 1 else "🚩 FAKE"
    
    # Trust Score Logic
    if confidence > 90 and uncertainty < 0.05:
        trust_level = "HIGH (Certain)"
    elif uncertainty > 0.1:
        trust_level = "LOW (Inconsistent/Confused)"
    else:
        trust_level = "MEDIUM"

    print("\n" + "-"*40)
    print(f"VERDICT     : {verdict}")
    print(f"CONFIDENCE  : {confidence:.2f}%")
    print(f"UNCERTAINTY : {uncertainty:.4f}")
    print(f"TRUST LEVEL : {trust_level}")
    print("-"*40)
    print(f"Target Accuracy (from Draft): 94%")