import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, Sigmoid

with open("labeled_result_corrected.json", "r") as file:
    data = json.load(file)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenized_data = []
dynamic_labels = []

for paper, details in data.items():
    concatenated_input = paper + " " + details["abstract"] + " " + " ".join(details["references"])
    inputs = tokenizer(concatenated_input, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    tokenized_data.append(inputs)
    
    label = [1 if ref in details["label"] else 0 for ref in details["references"]]
    dynamic_labels.append(label)

input_ids = [item["input_ids"].squeeze(0) for item in tokenized_data]
attention_masks = [item["attention_mask"].squeeze(0) for item in tokenized_data]

train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, dynamic_labels, random_state=42, test_size=0.1)
train_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=42, test_size=0.1)

def custom_collate_fn(batch):
    sorted_batch = sorted(batch, key=lambda x: len(x[2]), reverse=True)
    input_ids, attention_masks, labels = zip(*sorted_batch)
    max_len = len(labels[0])
    padded_labels = [label + [0] * (max_len - len(label)) for label in labels]
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    padded_labels = torch.tensor(padded_labels)
    return input_ids, attention_masks, padded_labels

batch_size = 8
train_dataloader = DataLoader(list(zip(train_inputs, train_masks, train_labels)), shuffle=True, batch_size=batch_size, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(list(zip(val_inputs, val_masks, val_labels)), shuffle=True, batch_size=batch_size, collate_fn=custom_collate_fn)

#model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
#max_num_references = max([len(refs) for refs in dynamic_labels])
max_num_references = 100
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=100)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = BCEWithLogitsLoss()

epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        batch_input_ids = batch[0]
        batch_input_mask = batch[1]
        batch_labels = batch[2]

        outputs = model(input_ids=batch_input_ids, attention_mask=batch_input_mask)
        logits = outputs[0] 
        batch_labels_padded = torch.nn.functional.pad(batch_labels, (0, max_num_references - batch_labels.shape[1]))

        loss = criterion(logits, batch_labels_padded.type_as(logits))
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch: {epoch+1}, Train Loss: {avg_train_loss:.4f}")

    model.eval()
    val_loss = 0
    correct_predictions = 0
    sigmoid = Sigmoid()

    with torch.no_grad():
        for batch in val_dataloader:
            batch_input_ids = batch[0]
            batch_input_mask = batch[1]
            batch_labels = batch[2]

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_input_mask)
            logits = outputs[0] 
            batch_labels_padded = torch.nn.functional.pad(batch_labels, (0, max_num_references - batch_labels.shape[1]))

            loss = criterion(logits, batch_labels_padded.type_as(logits))
            val_loss += loss.item()

            preds = sigmoid(logits)
            preds = (preds > 0.5).float()

            correct_predictions += torch.sum(preds == batch_labels_padded)

    avg_val_loss = val_loss / len(val_dataloader)
    val_accuracy = correct_predictions.double() / (len(val_dataloader) * batch_size * len(data))
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

torch.save(model.state_dict(), "custom_bert_model.pth")
