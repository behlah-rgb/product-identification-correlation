"""
Put the team's GitHub URL into the setup cell of every notebook (needed for Colab).

    python tools/set_repo_url.py https://github.com/<you>/product-identification-correlation.git
"""
import json
import re
import sys
from pathlib import Path

url = sys.argv[1]
for nb_path in sorted((Path(__file__).resolve().parent.parent / "notebooks").glob("*.ipynb")):
    nb = json.loads(nb_path.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = [re.sub(r'^REPO_URL = ".*"', f'REPO_URL = "{url}"', line) for line in cell["source"]]
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("updated", nb_path.name)
