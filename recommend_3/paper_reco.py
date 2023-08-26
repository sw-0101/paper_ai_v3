import torch
from transformers import BertTokenizer, BertModel
from torch.optim import Adam
from torch.nn import MSELoss

class PaperRecommender:
    def __init__(self, df):
        self.all_paper_titles = df['Title'].tolist()
        #self.all_paper_links = df['Link'].tolist()
        #self.query = query
        #self.all_paper_titles = all_paper_titles
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.regression_head = torch.nn.Linear(768, 10).to(self.device)
        self.optimizer = Adam(list(self.model.parameters()) + list(self.regression_head.parameters()), lr=1e-5)
        self.loss_function = MSELoss()

    def recommend_papers(self, query):
        inputs = self.tokenizer(' '.join(self.all_paper_titles) + ' ' + query[0], return_tensors="pt", truncation=True, padding=True, max_length=1024).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state.mean(dim=1)
            scores = self.regression_head(outputs).squeeze()
        recommended_indices = torch.topk(scores, 5).indices
        return recommended_indices.cpu(), scores.cpu()

    def train_model(self, query, feedback):
        inputs = self.tokenizer(' '.join(self.all_paper_titles) + ' ' + query[0], return_tensors="pt", truncation=True, padding=True, max_length=1024).to(self.device)
        outputs = self.model(**inputs).last_hidden_state.mean(dim=1)
        scores = self.regression_head(outputs).squeeze()

        feedback_tensor = torch.tensor(feedback, dtype=torch.float32).to(self.device)
        loss = self.loss_function(scores, feedback_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run_iterations(self, query, num_iterations=2):
        for i in range(num_iterations):
            recommended_indices, scores = self.recommend_papers(query)
            #print("Recommended paper indices:", recommended_indices.tolist())
            #print("Scores:", scores.tolist())
            feedback_for_all_papers = [0,1,0,1,0,0,0,0,0,0]
            self.train_model(query, feedback_for_all_papers)

        #print(self.all_paper_titles)
        #recommended_titles = [self.all_paper_titles[i] for i in recommended_indices.tolist()]
        #recommendations = {"titles": recommended_titles, "links": recommended_links}
        #return recommendations
        #return recommended_titles
        return recommended_indices
    #def inference_train(self, query):
        

# recommender = PaperRecommender(df)
# recommender.run_iterations(["segmentation"])
