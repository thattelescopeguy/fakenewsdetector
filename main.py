import os
import pandas as pd
fake=pd.read_csv(os.path.join("News_dataset/Fake.csv"))
true=pd.read_csv(os.path.join("News_dataset/True.csv"))
fake["label"]=0
true["label"]=1
data=pd.concat([fake, true], axis=0)
data=data.sample(frac=1).reset_index(drop=True)
data.to_csv("News_dataset/combined.csv", index=False)

from transformers import BertTokenizer, BertForSequenceClassification 
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report 
X=data['text'].tolist()
y=data['label'].tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings=tokenizer(X_train, truncation=True, padding=True, max_length=256)
test_encodings=tokenizer(X_test, truncation=True, padding=True, max_length=256)
###############################################################################
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)
train_dataset=NewsDataset(train_encodings, y_train)
test_dataset=NewsDataset(test_encodings, y_test)
model=BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
from torch.utils.data import DataLoader
from torch.optim import AdamW
train_loader=DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=8)
optim=AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(1):
    model.train()
    total_loss = 0 
    for step, batch in enumerate(train_loader):
        optim.zero_grad()
        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        labels=batch['labels'].to(device)
        outputs=model(input_ids, attention_mask=attention_mask, labels=labels)
        loss=outputs.loss
        total_loss+=loss.item()
        loss.backward()
        optim.step()
        if step % 100 == 0:  # print every 100 batches
            print(f"Epoch {epoch+1}, Step {step}, Batch Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} finished, Average Loss: {avg_loss:.4f}")
    model.save_pretrained(f"/content/drive/MyDrive/news/saved_model_epoch{epoch+1}")
    tokenizer.save_pretrained(f"/content/drive/MyDrive/news/saved_model_epoch{epoch+1}")
model.eval()
all_preds=[]
all_labels=[]
with torch.no_grad():# no gradients needed during evaluation
    for batch in test_loader:
        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        labels=batch['labels'].to(device)
        outputs=model(input_ids, attention_mask=attention_mask)
        logits=outputs.logits
        preds=torch.argmax(logits, dim=1)#pick class with highest score
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
accuracy = accuracy_score(all_labels, all_preds)
print("Test Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(all_labels, all_preds))
