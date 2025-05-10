import json
import re

def fix_trailing_comma(json_str):
    # Replace problematic trailing commas before closing brackets or braces
    fixed_str = re.sub(r',\s*}', '}', json_str)
    fixed_str = re.sub(r',\s*\]', ']', fixed_str)
    return fixed_str

def str2json(s):
    # Extract content between code fences
    pattern = r'```(?:json)?\n([\s\S]*?)\n```'
    match = re.search(pattern, s)
    
    if match:
        json_str = match.group(1)
        # Fix trailing commas
        json_str = fix_trailing_comma(json_str)
        try:
            # Parse the JSON string
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return s
    return s

# def str2json(summary: str) -> str:
#     try:
#         clean_summary = summary.replace("```json", "").replace("```", "").strip()
#         # clean_summary = clean_summary.replace("\n", '')
#         clean_summary = json.loads(clean_summary)
#     except:
#         clean_summary = summary
#     return clean_summary

def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
