import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import warnings

warnings.filterwarnings('ignore')

# --- LIGHTWEIGHT DATASET ---
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Using max_length=64 saves your CPU from crashing
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=64, 
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# --- MODEL ARCHITECTURE ---
class HybridTruthGuard(nn.Module):
    def __init__(self):
        super(HybridTruthGuard, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fusion = nn.Linear(768 + 64, 128) # BERT + Placeholder GCN
        self.classifier = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, ids, mask):
        outputs = self.bert(ids, mask)
        bert_feats = outputs.last_hidden_state[:, 0, :]
        gcn_placeholder = torch.zeros(bert_feats.shape[0], 64).to(bert_feats.device)
        combined = torch.cat([bert_feats, gcn_placeholder], dim=-1)
        return self.classifier(self.dropout(torch.relu(self.fusion(combined))))

# --- MAIN EXECUTION ---
def train_safely():
    device = 'cpu' # Forcing CPU to stay stable
    print(f"🚀 Starting Survival Training on {device}...")
    
    df = pd.read_csv('master_dataset_2026.csv')
    
    # --- CRITICAL: Training on 1,000 rows so your laptop stays cool ---
    df = df.sample(1000, random_state=42) 
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, stratify=df['label']
    )
    
    # Batch size 4 is very easy on RAM
    train_loader = DataLoader(FakeNewsDataset(X_train, y_train), batch_size=4, shuffle=True)
    test_loader = DataLoader(FakeNewsDataset(X_test, y_test), batch_size=4)

    model = HybridTruthGuard().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    print("\n[Training Epoch 1/1] This will take ~5 mins...")
    model.train()
    for batch in train_loader:
        ids, mask, lbls = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
        outputs = model(ids, mask)
        loss = criterion(outputs, lbls)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # --- EVALUATION ---
    model.eval()
    preds, actual = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            actual.extend(batch['label'].cpu().numpy())

    print("\n" + "="*35)
    print(f"✅ SUCCESS! Results for TruthGuard:")
    print(f"Accuracy:  {accuracy_score(actual, preds):.4f}")
    print(f"F1-Score:  {f1_score(actual, preds, average='weighted'):.4f}")
    print("="*35)
    
    torch.save(model.state_dict(), 'hybrid_model.pth')
    print("✓ Model saved safely as hybrid_model.pth")

if __name__ == '__main__':
    train_safely()