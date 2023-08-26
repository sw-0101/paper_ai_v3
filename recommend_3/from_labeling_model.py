#from api import df
import torch
from transformers import BertTokenizer, BertModel
from torch.optim import Adam
from torch.nn import MSELoss



all_paper_titles = df['Title'].tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)  
regression_head = torch.nn.Linear(768, 10).to(device)  
optimizer = Adam(list(model.parameters()) + list(regression_head.parameters()), lr=1e-5)
loss_function = MSELoss()

def recommend_papers(papers, query):
    inputs = tokenizer(' '.join(papers) + ' ' + query[0], return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)  # Move inputs to GPU
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)
        scores = regression_head(outputs).squeeze()
    recommended_indices = torch.topk(scores, 5).indices
    return recommended_indices.cpu(), scores.cpu() 

def train_model(papers, query, feedback):
    inputs = tokenizer(' '.join(papers) + ' ' + query[0], return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)  # Move inputs to GPU
    outputs = model(**inputs).last_hidden_state.mean(dim=1)
    scores = regression_head(outputs).squeeze()

    feedback_tensor = torch.tensor(feedback, dtype=torch.float32).to(device)  
    loss = loss_function(scores, feedback_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

query = ["segmentation"]
num_iterations = 3

for i in range(num_iterations):
    recommended_indices, scores = recommend_papers(all_paper_titles, query)
    print("Recommended paper indices:", recommended_indices.tolist())
    print("Scores:", scores.tolist())
    
    feedback = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    train_model(all_paper_titles, query, feedback)