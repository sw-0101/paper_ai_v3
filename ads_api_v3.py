import requests
import time
import json
import os

class get_references:
    
    def __init__(self, paper_title):
        self.queries = [paper_title]

    def search_nasa_ads(self, api_key, query):
        base_url = "https://api.adsabs.harvard.edu/v1/search/query?"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        params = {
            "q": query,
            "fl": "title,reference,abstract",
            "rows": 1  
        }

        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()["response"]["docs"][0]
            references = data.get("reference", [])
            reference_titles = []

            for ref_id in references:
                ref_query = f"arXiv:{ref_id}"
                ref_params = {
                    "q": ref_query,
                    "fl": "title",
                    "rows": 1
                }
                ref_response = requests.get(base_url, headers=headers, params=ref_params)
                if ref_response.status_code == 200:
                    ref_data = ref_response.json()["response"]["docs"][0]
                    reference_titles.append(ref_data.get("title", ["N/A"])[0])
                time.sleep(1)

            paper_title = data.get("title", ["N/A"])[0]
            paper_abstract = data.get("abstract", "N/A")
            result = {
                paper_title: {
                    "references": reference_titles,
                    "abstract": paper_abstract,
                }
            }

            return result
        else:
            return None

    def extraction(self):
        api_key = "INRAyIJJ6UyDcsyvIsP08nB8r0v4p7yXOARw9upE"  
        #print(self.queries)
        for query in self.queries:
            result = self.search_nasa_ads(api_key, query)
            if result:
                return result #json

# refer = get_references("Attention Is All You Need")
# print(refer.extraction())