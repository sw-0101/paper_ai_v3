import requests
from bs4 import BeautifulSoup
from scholarly import scholarly
import time

def get_references_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    all_references = []
    while url:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extracting references from the current page
        references_divs = soup.find_all("div", class_="gs_ri")
        references = [ref_div.h3.a.text for ref_div in references_divs]
        all_references.extend(references)

        # Looking for the "Next" link to navigate to the next page of references
        next_button = soup.find("button", attrs={"aria-label": "Next"})
        if next_button and "disabled" not in next_button.attrs:
            # Click the next button by simulating the behavior
            start_value = soup.find("input", {"name": "start"}).get("value")
            url = f"https://scholar.google.com/scholar?start={start_value}&hl=en&as_sdt=5,33&sciodt=0,33&cites=3927741869042927631&scipsc="
        else:
            url = None

        # Sleep for a bit to avoid hitting rate limits
        time.sleep(1)

    return all_references


def get_paper_details(paper_title):
    search_query = scholarly.search_pubs(paper_title)
    paper = next(search_query)
    return paper

def main(paper_title):
    paper_data = get_paper_details(paper_title)
    
    # Extracting basic details from the paper data
    title = paper_data['bib'].get("title")
    authors = paper_data['bib'].get("author")
    pub_year = paper_data['bib'].get("pub_year")
    abstract = paper_data['bib'].get("abstract")

    print(f"Title: {title}")
    print(f"Authors: {authors}")
    print(f"Year: {pub_year}")
    print(f"Abstract: {abstract}")
    print("-" * 50)

    # Fetching the list of papers that cite the given paper
    citedby_url = 'https://scholar.google.com' + paper_data['citedby_url']
    citations = get_references_from_url(citedby_url)

    print("List of papers that cite the given paper:")
    for i, citation in enumerate(citations, 1):
        print(f"{i}. {citation}")

if __name__ == "__main__":
    paper_title_input = "Fast Segment Anything"
    main(paper_title_input)
