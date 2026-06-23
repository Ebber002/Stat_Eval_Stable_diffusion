import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import glob
import os

CSV_FILE = "prompts/prompts_english.csv"
IMAGE_DIR = "outputs/output_english"
OUTPUT_CSV = "evaluated_results_clip.csv"
LANGUAGE = "ENG"

model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

df = pd.read_csv(CSV_FILE)
df["CLIP_Score"] = None

for index, row in df.iterrows():
    if row["Evaluation_Type"] == "Bias":
        continue
    
    img_id = str(row["ID"]).zfill(3)
    subject = row["Subject"]
    eval_type = row["Evaluation_Type"]
    prompt_text = row["Prompt"]

    search = os.path.join(IMAGE_DIR, f"{LANGUAGE}_{img_id}_{subject}_{eval_type}*.png") # <-- add LANGUAGE at before {img_id} for english images
    matched_files = glob.glob(search)

    if not matched_files:
        continue

    image_path = matched_files[0]

    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=[prompt_text], images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        cosine_score = F.cosine_similarity(image_embeds, text_embeds).item()

        df.at[index, "CLIP_Score"] = round(cosine_score, 4)
    print(f"[{img_id}/600]")
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print("Done")
