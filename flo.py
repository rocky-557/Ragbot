import json
from vec_db import vecdb
from model import gen_response
def gen_flo(query):
    res = vecdb.similarity_search(query)
    prompt = "Reformat the given context suitable for creating a flowchart as JSON with nodes and edges as keys (reply only flowchart data)"
    resp = gen_response(context=res, query=prompt, maxx=700)
    return resp

def genn(query):
    k = gen_flo(query).split('\n')
    json_string = ''.join(k).replace('```json', '').replace('```', '').strip()
    
    try:
        flowchart = json.loads(json_string)
        return flowchart
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
