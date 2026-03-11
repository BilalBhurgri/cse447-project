import requests
import time
from pathlib import Path

wiki_languages = [
    "en", "zh", "hi", "es", "fr", "ar", "bn",
    "pt", "ru", "ur", "id", "de", "ja", "sw", "mr"
]

num_examples = 5000
chunk_size = 1024

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "CSE447-Wikipedia-Dataset-Collector/1.0 (bilal)"
}

def get_random_pages(lang, n=50, retries=5):
    url = f"https://{lang}.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": n,
        "prop": "extracts",
        "explaintext": True,
    }

    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)

            if r.status_code != 200:
                raise Exception(f"Bad status: {r.status_code}")

            data = r.json()

            pages = []
            if "query" in data:
                for p in data["query"]["pages"].values():
                    pages.append(p.get("extract", ""))

            return pages

        except Exception as e:
            print(f"{lang}: API error ({e}), retry {attempt+1}/{retries}")
            time.sleep(2)

    return []


def collect_examples(lang):
    examples = []

    print(f"\nCollecting data for {lang}...")

    while len(examples) < num_examples:
        pages = get_random_pages(lang, 500)

        for text in pages:
            text = text.replace("\n", " ").strip()

            if len(text) < chunk_size:
                continue

            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                examples.append(chunk)

                if len(examples) >= num_examples:
                    break

            if len(examples) >= num_examples:
                break

        print(f"{lang}: {len(examples)}/{num_examples}")

    out_file = output_dir / f"{lang}.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex + "\n")

    print(f"{lang}: saved {len(examples)} examples")


def main():
    for lang in wiki_languages:
        collect_examples(lang)


if __name__ == "__main__":
    main()