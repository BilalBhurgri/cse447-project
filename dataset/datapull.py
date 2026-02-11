from datasets import load_dataset
import unicodedata

languages = ["en_us",
             "cmn_hans_cn",
    "hi_in",
    "es_419",
    "fr_fr",
    "ar_eg",
    "bn_in",
    "pt_br",
    "ru_ru",
    "ur_pk",
    "id_id",
    "de_de",
    "ja_jp",
    "sw_ke",
    "mr_in",]

def normalize(example):
    text = example.get("transcription")
    if text is not None:
        text = unicodedata.normalize("NFKC", text)
    normalized_example = {
        "text": text,
        "language": example["language"], 
    }
    return normalized_example

datasets = {}
MAX_EXAMPLES = 50000

for lang in languages:
    print(f"Loading {lang}...")
    ds = load_dataset(
        "google/fleurs",
        lang,
        split="train",
    )

    print(f"Limiting {lang} dataset to {MAX_EXAMPLES} examples...")
    ds = ds.select(range(min(len(ds), MAX_EXAMPLES)))

    print(f"Normalizing {lang} dataset...")
    ds = ds.map(normalize)

    print(f"Filtering {lang} dataset for missing text...")
    ds = ds.filter(lambda x: x["text"] is not None)

    print(f"Number of examples in {lang} after filtering: {len(ds)}")

    datasets[lang] = ds

    savepath = f"fleurs{lang}_{MAX_EXAMPLES}"
    print(f"\nSaving the dataset for {lang} locally to {savepath}...")
    ds.save_to_disk(savepath)

for lang in languages:
    print(f"\n--- {lang} samples ---")
    for i, example in enumerate(datasets[lang]):
        print(f"Sample {i + 1}: {example}")
        if i >= 2: 
            break