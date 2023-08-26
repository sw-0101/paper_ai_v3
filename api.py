import os
import sys
import urllib.request
import json
import re
import requests
import simplejson
import pandas as pd
from datetime import datetime
import time
import asyncio
import websockets
import json
from paper_reco import PaperRecommender
from ads_api_v2 import get_references
from inference import PaperImportancePredictor

Google_SEARCH_ENGINE_ID = "a349af633f7cb4d38"
Google_API_KEY = "AIzaSyAeScroDOok-8HcoT8r-6_mCbMJRDI4Ex0"

Search_Link = ['arxiv']    
title_page = []


def Google_API(query, wanted_row):
    query= query.replace("|","OR")
    #query += "-filetype:pdf"
    start_pages=[]

    df_google= pd.DataFrame(columns=['Title','Link']) 
    row_count = 0

    for i in range(1,wanted_row+1000,10):
        start_pages.append(i)

    for start_page in start_pages:
        url = f"https://www.googleapis.com/customsearch/v1?key={Google_API_KEY}&cx={Google_SEARCH_ENGINE_ID}&q={query}&start={start_page}"
        #print(url)
        res = requests.get(url)
        data = res.json()
        search_items = data.get("items")

        try:
            for i, search_item in enumerate(search_items, start=1):
                # extract the page url
                link = search_item.get("link")
                #print(link)
                if any(r in link for r in Search_Link): #특정 conference에서만
                    # get the page title
                    title = search_item.get("title")
                    # page snippet
                    #descripiton = search_item.get("snippet")
                    # print the results
                    df_google.loc[start_page + i] = [title,link] #description
                    row_count += 1
                    if (row_count >= wanted_row) or (row_count == 300) :
                        return df_google
                else:
                    pass
        except:
            return df_google
    return df_google

def final(query, wanted_row=10):
    df_google=Google_API(query, wanted_row)
    #df_google['search_engine'] = 'Google'
    df_result = pd.concat([df_google])
    #today = datetime.today().strftime("%Y%m%d")
    #df_result['search_date'] = today
    df_result.reset_index(inplace=True, drop=True)
    return df_result

# query = "Segment" 
# df = final(query = query, wanted_row=10)
# model = PaperRecommender(df)
# recommend_papers = model.run_iterations(query)
# print(recommend_papers)

query = "Segment" #user query
df = final(query=query, wanted_row=10)
model = PaperRecommender(df)
recommend_papers_indices = model.run_iterations(query)
recommended_titles = df['Title'].iloc[recommend_papers_indices].tolist()
recommended_links = df['Link'].iloc[recommend_papers_indices].tolist()

recommended_dict = {
    "Title": recommended_titles,
    "Link": recommended_links
}
refers = get_references(recommended_dict["Title"][0])
refers_paper0 = refers.extraction()
predictor = PaperImportancePredictor(model_path="custom_bert_model.pth")

topk_titles = predictor.get_topk_references(refers_paper0)
#print(topk_titles)
tree_list = []
for title in topk_titles:
    refers = get_references(title)
    refers_paper = refers.extraction()
    #print(predictor.get_topk_references(refers_paper))
    tree_list.append(title)
    for i in predictor.get_topk_references(refers_paper):
        tree_list.append(i)

#print(tree_dict)

# with open("recommended_with_references.json", "w") as outfile:
#     json.dump(refers_paper, outfile, indent=4)


#recommended_dict["TItle"][0]

# async def accept(websocket, path):
#     print(path)
#     while True: 
#         data = await websocket.recv()
#         searchdata = data.split(" ")
#         conference = searchdata[0]
#         year = searchdata[1]
#         query = searchdata[2]
#         Search_Link.append(conference)
#         print(query)
#         print(Search_Link)
        
#         await websocket.send()

# async def serve():
#     server = await websockets.serve(accept, "localhost", 9998)
#     await server.wait_closed()

# asyncio.run(serve())