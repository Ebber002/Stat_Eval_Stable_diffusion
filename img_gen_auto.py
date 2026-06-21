import pandas as pd
import json
import urllib.request
import urllib.parse
import time

CSV_FILE = "prompts\prompts_english.csv"
LANGUAGE = "ENG"
WORKFLOW_FILE = "workflows\workflow1.json"
COMFY_URL = "http://127.0.0.1:8188/prompt"

PROMPT_NODE_ID = "3"
SEED_NODE_ID = "2"
SAVE_NODE_ID = "6"

def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(COMFY_URL, data=data)
    response = urllib.request.urlopen(req)
    return json.loads(response.read())

df = pd.read_csv(CSV_FILE)
with open(WORKFLOW_FILE, "r", encoding="utf-8") as f:
    workflow = json.load(f)

for index, row in df.iterrows():
    img_id = str(row["ID"]).zfill(3)
    subject = row["Subject"]
    eval_type = row["Evaluation_Type"]
    prompt_text = row["Prompt"]
    seed = row["Seed"]

    workflow[PROMPT_NODE_ID]["inputs"]["text"] = prompt_text
    workflow[SEED_NODE_ID]["inputs"]["seed"] = seed
    
    filename = f"{LANGUAGE}_{img_id}_{subject}_{eval_type}"
    workflow[SAVE_NODE_ID]["inputs"]["filename_prefix"] = filename

    result = queue_prompt(workflow)
    print(f"[{img_id}/600]: {subject}. {eval_type}. Seed: {seed}")
    time.sleep(0.1)

print("Done")