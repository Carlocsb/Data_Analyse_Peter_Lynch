from utils import es_client
es = es_client()

index = "stocks"

resp = es.search(
    index=index,
    body={
        "size": 1,
        "_source": True,
        "query": {
            "bool": {
                "must": [
                    {"term": {"symbol.keyword": "BNL"}},
                    {"term": {"date": "2024-09-30"}},
                ]
            }
        }
    }
)

print("Hits:", len(resp["hits"]["hits"]))
if resp["hits"]["hits"]:
    from pprint import pprint
    pprint(resp["hits"]["hits"][0]["_source"])
