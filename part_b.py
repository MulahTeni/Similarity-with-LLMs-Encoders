import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# MODEL adı
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

# Cihaz ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model ve tokenizer yükle
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Pooling fonksiyonu
def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# Vektör çıkarım fonksiyonu
def get_embedding(texts):
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**batch)
        embeddings = average_pool(outputs.last_hidden_state, batch["attention_mask"])
        embeddings = embeddings.cpu().numpy()
    return embeddings

# Veri yükleme ve ilk 1000 satır seçimi
df = pd.read_csv("sc.csv").dropna()
df.columns = ["question", "gpt4o", "deepseek", "label"]
df["label"] = df["label"].astype(int)
df = df.iloc[:1000]

# Embedding çıkarımı
print("Embedding çıkarılıyor...")
s_emb = get_embedding(df["question"].tolist())
g_emb = get_embedding(df["gpt4o"].tolist())
d_emb = get_embedding(df["deepseek"].tolist())

# Vektör kombinasyonları (örnek: s-g, s-d, |s-g| - |s-d|)
features = np.hstack([
    np.abs(s_emb - g_emb),               # |s - g|
    np.abs(s_emb - d_emb),               # |s - d|
    np.abs(np.abs(s_emb - g_emb) - np.abs(s_emb - d_emb))  # | |s-g| - |s-d| |
])

# Eğitim/test ayırımı
X_train, X_test, y_train, y_test = train_test_split(features, df["label"], test_size=0.2, random_state=42)

# Model eğitimi
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Değerlendirme
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
