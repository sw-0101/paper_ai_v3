import requests
from paper_reco import PaperRecommender
import requests

def get_arxiv_papers(query, max_results=10):
    base_url = 'http://export.arxiv.org/api/query?'
    params = {
        'search_query': f'all:{query}',
        'start': 0,
        'max_results': max_results
    }
    response = requests.get(base_url, params=params)
    entries = response.text.split('<entry>')
    papers = []

    for entry in entries[1:]:
        title_start = entry.find('<title>') + 7
        title_end = entry.find('</title>')
        title = entry[title_start:title_end].strip()
        link_start = entry.find('<link href="') + 12
        link_end = entry.find('" rel="alternate"')
        link = entry[link_start:link_end].strip()
        
        papers.append({'title': title, 'link': link})

    return papers

query = ["Segment"] 
papers = get_arxiv_papers(query)
print(papers)
paper_title = [paper['title'] for paper in papers]
model = PaperRecommender(papers)
recommend_papers = model.run_iterations(query)

