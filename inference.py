import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from ads_api_v2 import get_references

class PaperImportancePredictor:
    def __init__(self, model_path, tokenizer_name='bert-base-uncased', model_name='bert-base-uncased', num_labels=100):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict_importance(self, paper_title, abstract, references):
        concatenated_input = paper_title + " " + abstract + " " + " ".join(references)
        inputs = self.tokenizer(concatenated_input, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs[0]
            scores = torch.sigmoid(logits).squeeze().tolist()

        return scores

    def get_topk_references(self, data, topk=2):
        title = list(data.keys())[0]
        paper_details = data.get(title, None)

        if paper_details:
            abstract = paper_details["abstract"]
            references = paper_details["references"]
            scores = self.predict_importance(title, abstract, references)
            sorted_references_scores = sorted(zip(references, scores), key=lambda x: x[1], reverse=True)
            topk_titles = [item[0] for item in sorted_references_scores[:topk]]
            return topk_titles
        else:
            return []

# if __name__ == "__main__":
#     predictor = PaperImportancePredictor(model_path="custom_bert_model.pth")
#     with open("recommended_with_references.json", "r") as file:
#         data = json.load(file)
#     title = list(data.keys())[0]
#     topk_titles = predictor.get_topk_references(data)
#     #print(topk_titles)
#     tree_dict = []
#     for title in topk_titles:
#         refers = get_references(title)
#         refers_paper = refers.extraction()
#         #print(predictor.get_topk_references(refers_paper))
#         #tree_dict[title] = predictor.get_topk_references(refers_paper)
#         tree_dict.append(title)
        
#         for i in predictor.get_topk_references(refers_paper):
#             tree_dict.append(i)


#     print(tree_dict)
