import json
import sys
from pathlib import Path

def rewrite_images_prefix(json_file: str, new_prefix: str):
    """
    json_file : str  the path of the JSON file to be modified
    new_prefix: str  the new prefix that will replace /your_root_path
    """
    old_prefix = "/your_root_path"

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if "images" in item and isinstance(item["images"], list):
            new_images = []
            for path in item["images"]:
                if isinstance(path, str) and path.startswith(old_prefix):
                    new_images.append(str(Path(new_prefix) / Path(path).relative_to(old_prefix)))
                else:
                    new_images.append(path)
            item["images"] = new_images

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # how to use：python script.py data.json /new/prefix
    rewrite_images_prefix(sys.argv[1], sys.argv[2])