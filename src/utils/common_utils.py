import json

def str2json(summary: str) -> str:
    try:
        clean_summary = summary.replace("```json", "").replace("```", "").strip()
        clean_summary = json.loads(clean_summary)
    except:
        clean_summary = summary
    return clean_summary

def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
