import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_embeddings(model_name, texts):
    if model_name == "jinaai/jina-embeddings-v3":
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        embeddings = model.encode(texts, task="text-matching")
        return embeddings

    elif model_name == "intfloat/multilingual-e5-large-instruct":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        task_description = "Represent this sentence for retrieval:"
        inputs = [f"Instruct: {task_description}\nQuery: {t}" for t in texts]

        encoded = tokenizer(inputs, max_length=512, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = model(**encoded)
        embeddings = average_pool(output.last_hidden_state, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
        return embeddings

    elif model_name == "ytu-ce-cosmos/turkish-e5-large":
        model = SentenceTransformer(model_name)
        return model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)

    else:
        raise ValueError(f"Desteklenmeyen model: {model_name}")


def evaluate_top_k(soru_embeds, answer_embeds, k=5):
    sims = cosine_similarity(soru_embeds, answer_embeds)
    top_k = np.argsort(-sims, axis=1)[:, :k]
    corrects = np.arange(len(soru_embeds))

    top1_acc = np.mean(top_k[:, 0] == corrects)
    top5_acc = np.mean([corrects[i] in top_k[i] for i in range(len(corrects))])
    return top1_acc, top5_acc


def main():
    models = {
        "e5": "intfloat/multilingual-e5-large-instruct",
        "cosmosE5": "ytu-ce-cosmos/turkish-e5-large",
        "jina": "jinaai/jina-embeddings-v3"
    }

    df = pd.read_csv("sc.csv")
    df_sampled = df.sample(n=1000, random_state=42).reset_index(drop=True)

    all_results = []

    for tag, model_name in models.items():
        print(f"\n🚀 {tag.upper()} modeli işleniyor: {model_name}")

        soru_embeds = get_embeddings(model_name, df_sampled["soru"].tolist())
        gpt_embeds = get_embeddings(model_name, df_sampled["gpt4o"].tolist())
        deep_embeds = get_embeddings(model_name, df_sampled["deepseek"].tolist())

        gpt_top1, gpt_top5 = evaluate_top_k(soru_embeds, gpt_embeds)
        deep_top1, deep_top5 = evaluate_top_k(soru_embeds, deep_embeds)

        gpt_diag = np.diag(cosine_similarity(soru_embeds, gpt_embeds))
        deep_diag = np.diag(cosine_similarity(soru_embeds, deep_embeds))
        df_sampled["gpt_score"] = gpt_diag
        df_sampled["deep_score"] = deep_diag

        corr = df_sampled[["gpt_score", "deep_score", "hangisi_iyi"]].corr()

        os.makedirs(f"model_output_{tag}", exist_ok=True)
        plt.figure(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"{tag.upper()} - Korelasyon Matrisi")
        plt.tight_layout()
        plt.savefig(f"model_output_{tag}/korelasyon.png")
        plt.close()

        all_results.append({
            "Model": tag,
            "GPT4o Top-1": gpt_top1,
            "GPT4o Top-5": gpt_top5,
            "DeepSeek Top-1": deep_top1,
            "DeepSeek Top-5": deep_top5
        })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("tum_model_sonuclari.csv", index=False)
    print("\n✅ Tüm modeller işlendi ve başarılar 'tum_model_sonuclari.csv' olarak kaydedildi.")


if __name__ == "__main__":
    main()
