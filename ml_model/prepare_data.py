import pandas as pd
import os

def load_smart(filename, default_label=None):
    full_path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(full_path):
        print(f"⚠️ Skipping {filename}: Not found.")
        return pd.DataFrame()
    
    print(f"Reading {filename}...")
    try:
        if filename.endswith('.xlsx'):
            df = pd.read_excel(full_path, engine='openpyxl')
        else:
            df = pd.read_csv(full_path, on_bad_lines='skip', engine='python')

        # 1. Clean columns and reset index
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.reset_index(drop=True)

        # 2. Map the Label (Pick only the FIRST match)
        target_label = None
        if default_label is not None:
            target_label = pd.Series([default_label] * len(df))
        else:
            for col in ['label', 'verdict', 'target', 'class', 'status']:
                if col in df.columns:
                    # Explicitly take only the first column if multiples exist
                    target_label = df[col]
                    if isinstance(target_label, pd.DataFrame):
                        target_label = target_label.iloc[:, 0]
                    break
        
        # 3. Map the Text (Pick only the FIRST match)
        target_text = None
        # Prioritize English translation if it exists for your Bharat data
        for col in ['eng_trans_news body', 'news body', 'text', 'content', 'article', 'statement']:
            if col in df.columns:
                target_text = df[col]
                if isinstance(target_text, pd.DataFrame):
                    target_text = target_text.iloc[:, 0]
                break
        
        if target_text is not None and target_label is not None:
            temp = pd.DataFrame()
            temp['text'] = target_text.astype(str)
            temp['label'] = pd.to_numeric(target_label, errors='coerce')
            return temp.dropna()
        else:
            print(f"⚠️ Warning: Mapping failed for {filename}")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        return pd.DataFrame()

# --- RUN MERGE ---
files = [('Fake.csv', 0), ('True.csv', 1), ('WELFake_Dataset.csv', None), ('bharatfakenewskosh.xlsx', None)]
all_data = []

for f, l in files:
    d = load_smart(f, l)
    if not d.empty:
        all_data.append(d)

if all_data:
    final = pd.concat(all_data, ignore_index=True)
    final['label'] = final['label'].astype(int)
    final = final.sample(frac=1).reset_index(drop=True)
    final.to_csv('master_dataset_2026.csv', index=False)
    print("-" * 35)
    print(f"✅ REAL SUCCESS!")
    print(f"Total rows: {len(final)}")
    print("-" * 35)