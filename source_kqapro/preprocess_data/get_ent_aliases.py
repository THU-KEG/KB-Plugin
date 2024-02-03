from wikidata.client import Client
import json
from tqdm import tqdm
from parallel import parallel_process_data


client = Client()  # doctest: +SKIP

def get_aliases(id):
    tries = 0
    while tries < 5:
        tries += 1
        try:
            entity = client.get(id, load=True)
            aliases = [x["value"] for x in entity.data["aliases"].get("en", [])]
            if "en" in entity.data["labels"]:
                label = entity.data["labels"]["en"]["value"]
                if label not in aliases:
                    aliases = [label] + aliases
            return aliases
        except Exception as e:
            print(e)
            continue
    return []

data = {}

kb = json.load(open("data/kb.json"))
ents = []
for h_id, concept in tqdm(list(kb["concepts"].items())):
    ents.append((h_id, concept["name"]))

for h_id, ent in tqdm(list(kb["entities"].items())):
    ents.append((h_id, ent["name"]))

pbar = tqdm(total=len(ents))

def handle_item(ent):
    pbar.update(1)
    id, name = ent
    if id not in data:
        aliases = get_aliases(id)
        if name not in aliases:
            aliases = [name] + aliases
        data[id] = {
            "name": name,
            "aliases": aliases,
        }

parallel_process_data(ents, handle_item, workers=128)
json.dump(data, open("data/q_id2aliases.json", "w"), indent=2, ensure_ascii=False)
