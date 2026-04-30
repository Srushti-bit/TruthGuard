import torch
import pandas as pd
from torch.utils.data import DataLoader
from train_model import HybridTruthGuard, BertTextEncoder, TimeAwareGCN, FakeNewsDataset, evaluate
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os

def reload_and_report():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path = 'master_dataset_2026.csv'
    model_file = 'hybrid_model.pth'

    if not os.path.exists(model_file):
        print(f"❌ Error: {model_file} not found in this folder!")
        return

    print(f"🔄 Loading saved model: {model_file}...")
    
    # 1. Initialize architecture
    bert = BertTextEncoder()
    gcn = TimeAwareGCN()
    model = HybridTruthGuard(bert, gcn).to(device)
    
    # 2. Load the weights
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # 3. Load the data
    print(f"📂 Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Use a sample for quick calculation
    df = df.sample(min(1000, len(df)), random_state=42)
    
    _, X_test, _, y_test = train_test_split(
        df['text'].tolist(), df['label'].tolist(), 
        test_size=0.2, stratify=df['label'], random_state=42
    )
    
    test_loader = DataLoader(FakeNewsDataset(X_test, y_test), batch_size=8)
    criterion = nn.CrossEntropyLoss()

    # 4. Calculate Metrics
    print("📊 Calculating final metrics...")
    metrics = evaluate(model, test_loader, criterion, device)

    print("\n" + "="*40)
    print("         TRUTHGUARD FINAL REPORT")
    print("="*40)
    print(f"✅ Accuracy:  {metrics['acc']:.4f}")
    print(f"✅ F1-Score:  {metrics['f1']:.4f}")
    print(f"✅ Precision: {metrics['precision']:.4f}")
    print(f"✅ Recall:    {metrics['recall']:.4f}")
    print("="*40)

if __name__ == '__main__':
    reload_and_report()