
import requests

API_ENDPOINT = "https://api.webofscience.com/"  # Hypothetical endpoint; you need the actual endpoint
API_KEY = "https://api.clarivate.com/apis/wos-starter/v1"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def get_paper_uid_by_title(title):
    """Search for a paper by title and return its UID."""
    search_url = f"{API_ENDPOINT}/search"
    payload = {
        "query": title,
        # Additional search parameters here...
    }
    response = requests.post(search_url, headers=headers, json=payload)
    data = response.json()
    if 'records' in data:
        return data['records'][0]['UID']  # Hypothetically getting the first UID
    return None

def get_references_by_uid(uid):
    """Retrieve references for a paper by its UID."""
    retrieve_url = f"{API_ENDPOINT}/retrieve/{uid}"
    response = requests.get(retrieve_url, headers=headers)
    data = response.json()
    if 'references' in data:
        return [ref['title'] for ref in data['references']]
    return []

# Example
title = "Fast Segment Anything"
uid = get_paper_uid_by_title(title)
if uid:
    references = get_references_by_uid(uid)
    print(references)
else:
    print("Paper not found.")