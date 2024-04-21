import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_json('movies.json')

# Preprocessing text
def preprocess_text(s):
    s = re.sub(r'[^\w\s]', '', s)  # Remove punctuation
    s = s.lower()  # Lowercase text
    return s

df['description'] = df['description'].apply(preprocess_text)

# Filter the dataframe for only relevant genres for each iteration. We have ran the code with 2, 3, 4, 5 and all 19 genres.
df = df[df['genre'].isin(['Action', 'Drama', 'Comedy', 'Horror', 'Animation', 'Adventure', 'Thriller', 'Romance', 'Crime', 'Science Fiction', 'Family', 'Fantasy', 'Mystery', 'Documentary', 'Western', 'War', 'Music', 'History', 'TV Movie'])]

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(texts):
    return tokenizer(texts, add_special_tokens=True, max_length=120, padding='max_length', truncation=True, return_tensors='pt')

# Tokenize all descriptions
encoded_batch = tokenize_function(df['description'].tolist())
df['input_ids'] = [tensor.squeeze() for tensor in encoded_batch['input_ids']]
df['attention_mask'] = [tensor.squeeze() for tensor in encoded_batch['attention_mask']]

# Encode labels
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['genre'])

# Split data into training, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

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
val_dataset = MovieDataset(val_df)
test_dataset = MovieDataset(test_df)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 3)

best_model = None
best_val_loss = float('inf')

# Training and Validation
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_val_loss += loss.item()
    val_loss = total_val_loss / len(val_loader)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()

# Load best model for evaluation
model.load_state_dict(best_model)
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=-1).tolist())
        actuals.extend(batch['labels'].tolist())

# Calculate accuracy
accuracy = accuracy_score(actuals, predictions)
print(f'Accuracy: {accuracy:.4f}')  # Print the accuracy with 4 decimal places

# Generate classification report and modify index to show genre names
report = pd.DataFrame(classification_report(actuals, predictions, output_dict=True)).transpose()
report.drop(['accuracy'], inplace=True)  # Remove the 'accuracy' row if it's present
report['support'] = report['support'].apply(int)
# Map numeric labels back to string names using LabelEncoder
report.index = [label_encoder.inverse_transform([int(idx)])[0] if idx.isdigit() else idx for idx in report.index]

# Add a print for the classification report
print(report)

# Visualization of the Classification Report
fig, ax = plt.subplots(figsize=(8, 5))
report[['precision', 'recall', 'f1-score']].plot(kind='barh', ax=ax)
ax.text(0.45, 1.1, f'Accuracy: {accuracy:.2f}', transform=ax.transAxes)
ax.set_title('Classification Report')
ax.set_xlim([0, 1])
plt.show()

# Confusion Matrix with genre names
conf_mat = confusion_matrix(actuals, predictions)
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()