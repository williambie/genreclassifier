import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm

# Load data
df = pd.read_json('movies.json')

# Filter the dataframe for only Action and Drama genres
df = df[df['genre'].isin(['Action', 'Drama'])]

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to tokenize text
def tokenize_function(texts):
    return tokenizer(texts, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors='pt')

# Tokenize all descriptions
encoded_batch = tokenize_function(df['description'].tolist())
df['input_ids'] = [tensor.squeeze() for tensor in encoded_batch['input_ids']]
df['attention_mask'] = [tensor.squeeze() for tensor in encoded_batch['attention_mask']]

# Encode labels
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['genre'])

# Split data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.1)

# Create PyTorch dataset
class MovieDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries.iloc[idx]
        input_ids = entry['input_ids']
        attention_mask = entry['attention_mask']
        labels = torch.tensor(entry['labels'], dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

train_dataset = MovieDataset(train_df)
test_dataset = MovieDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 3)

# Training
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# Evaluation
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=-1).tolist())
        actuals.extend(batch['labels'].tolist())

accuracy = classification_report(actuals, predictions, target_names=label_encoder.classes_)
print("Classification Report:")
print(accuracy)
