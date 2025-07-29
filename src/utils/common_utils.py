import json
import re
import logging
logger = logging.getLogger(__name__)

def fix_trailing_comma(json_str):
    # Replace problematic trailing commas before closing brackets or braces
    fixed_str = re.sub(r',\s*}', '}', json_str)
    fixed_str = re.sub(r',\s*\]', ']', fixed_str)
    return fixed_str

def fix_json_string(json_str: str) -> str:
    """
    Cleans up a JSON string by fixing common LLM-generated errors,
    such as trailing commas and unescaped newlines within string values.
    """
    # 1. Fix trailing commas before brackets or braces
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

    # 2. Fix unescaped newlines inside string values.
    #    This is a two-step process:
    #    a) Define a function that takes a match object and escapes newlines
    #       ONLY within that match.
    #    b) Use re.sub to apply this function to all quoted string content.
    def escape_newlines_in_match(match):
        # group(0) is the entire matched string, including the quotes.
        # We replace \n with \\n inside this matched substring.
        return match.group(0).replace('\n', '\\n')

    # The regex r'"[^"\\]*(\\.[^"\\]*)*"' is a standard and robust pattern
    # for matching a complete double-quoted string, correctly handling
    # escaped quotes (\") inside it.
    # We apply our function to every string literal found.
    json_str = re.sub(r'"[^"\\]*(\\.[^"\\]*)*"', escape_newlines_in_match, json_str)
    
    return json_str

# The main str2json function remains the same, as it correctly calls fix_json_string.
def str2json(text: str) -> dict | list | str:
    """
    Extracts a JSON object or list from a string by finding all markdown
    code blocks and attempting to parse their contents. It is highly robust
    against common LLM formatting errors.
    """
    pattern = r'```json\n?([\s\S]*?)\n?```'
    matches = re.findall(pattern, text)

    if matches:
        for block_content in reversed(matches):
            potential_json = block_content.strip()
            try:
                # Apply all fixes *before* parsing
                fixed_json_str = fix_json_string(potential_json)
                return json.loads(fixed_json_str)
            except json.JSONDecodeError:
                continue

    try:
        fixed_text = fix_json_string(text)
        return json.loads(fixed_text)
    except json.JSONDecodeError:
        logger.error("Could not find/parse any valid JSON in the provided text.")
        return text

def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    test_str = """[
{
"Intent": "The core intent of the question is to identify the capital city of France."
}
]"""

    print(str2json(test_str))

