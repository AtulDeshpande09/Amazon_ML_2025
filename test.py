import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load subset
test_df = pd.read_csv("dataset/test.csv")
subset_df = test_df.head(25000).copy()

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
test_encodings = tokenizer(
    subset_df['catalog_content'].astype(str).tolist(),
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("./bert_price_model").to(device)
model.eval()

# Predict (batched)
batch_size = 16
preds_list = []
for i in range(0, len(subset_df), batch_size):
    batch = {k: v[i:i+batch_size].to(device) for k, v in test_encodings.items()}
    with torch.no_grad():
        outputs = model(**batch)
    preds_list.append(outputs.logits.squeeze().detach().cpu())

preds = torch.cat(preds_list).numpy()

# If trained on log(price):
pred_prices = np.expm1(preds)
#pred_prices = preds

# Submission
submission = pd.DataFrame({
    "sample_id": subset_df["sample_id"],
    "price": pred_prices
})

submission.to_csv("submission_partial.csv", sep="\t", index=False)
print(" Done! Saved partial predictions for", len(subset_df), "rows.")
