import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

data = pd.read_csv("dataset/train.csv", engine="python", on_bad_lines='skip')
data['catalog_content'] = data['catalog_content'].astype(str).fillna("")

print("CUDA available:", torch.cuda.is_available())

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

encodings = tokenizer(
    data["catalog_content"].tolist(),
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt"
)


# target value 
labels = np.log1p(data['price'].values)

# dataset
class PriceDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

dataset = PriceDataset(encodings, labels)


model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=1,
    problem_type="regression"
)

# training using trainer
training_args = TrainingArguments(
    output_dir="./bert_price_model",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    save_strategy="no",
    weight_decay=0.01,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# save model
trainer.save_model("./bert_price_model")
tokenizer.save_pretrained("./bert_price_model")

print(" Model & Tokenizer Saved Successfully!")
