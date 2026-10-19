"""Run in the original Colab session (after the notebook cells) to export
artifacts that only exist there.

1. Style drift values and the drift chain texts -> results/style_drift.csv
2. The 12 fine-tuned GPT-2 models -> Hugging Face Hub

Paste into a new cell at the end of the notebook. For step 2, add an HF_TOKEN
with write access to Colab secrets and set HF_USER below.
"""
import os

import pandas as pd

# 1. Style drift: `drift` comes from the notebook's drift cell.
#    If the runtime was restarted, rerun that cell first. Values will differ
#    slightly from the paper's figure because generation is sampled.
pd.DataFrame({
    "generation_step": list(range(1, len(drift) + 1)),
    "style_similarity": [float(x) for x in drift],
}).to_csv("style_drift.csv", index=False)
print("Wrote style_drift.csv (download it and put it in results/)")

# 2. Upload fine-tuned models
from google.colab import userdata
from huggingface_hub import HfApi

HF_USER = "your-hf-username"  # change this
api = HfApi(token=userdata.get("HF_TOKEN"))

for author in ["author_0", "author_1", "author_2", "author_3"]:
    for size in [25, 50, 100]:
        name = f"{author}_{size}"
        repo_id = f"{HF_USER}/gpt2-micro-style-{name.replace('_', '-')}"
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=os.path.join("/content/drive/MyDrive/micro_style_models", name),
        )
        print("Uploaded", repo_id)
