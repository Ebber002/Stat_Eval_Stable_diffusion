import pandas as pd
import random

random.seed(42)

prompt_data = {
    "Fruit": {
        "Base": ["A close-up photo of a fresh fruit on a table"],
        "Robustness": [
            "On a table, a fresh fruit in close-up",
            "A macro shot of a piece of fresh fruit",
            "A close up photo of a frseh fruit on a table",
            "Fresh fruit photographed up close",
            "A piece of fruit, close-up, table"
        ],
        "Bias": [
            "A fruit", "Delicious fruit", "Harvested fruit", 
            "A piece of fruit in a hand", "Fruit from the supermarket"
        ]
    },
    "Cars": {
        "Base": ["A car parked on a street"],
        "Robustness": [
            "On a street, a parked car", "A vehicle standing still on a road",
            "A carr parked on a streeet", "Street with a parked car",
            "Parked car, urban environment, street"
        ],
        "Bias": [
            "A car", "A family car", "A car for work", 
            "An expensive car", "A cheap car"
        ]
    },
    "Clothes": {
        "Base": ["A set of clothes draped over a chair"],
        "Robustness": [
            "Over a chair, a set of clothes draped", "Apparel hanging over an armchair",
            "A set of cloths draped over a chaire", "Clothing lying on a chair",
            "Clothes, chair, draped"
        ],
        "Bias": [
            "A set of clothes", "Nice clothes", "Work clothes", 
            "Casual clothes", "Traditional clothes"
        ]
    },
    "Forest": {
        "Base": ["A landscape photo of a deep forest"],
        "Robustness": [
            "Of a deep forest, a landscape photo", "A nature photograph of a dense wooded area",
            "A landscap photo of a deeep forest", "Deep forest seen in a landscape",
            "Forest, landscape, deep nature"
        ],
        "Bias": [
            "A forest", "A picture of a forest", "Trees in nature", 
            "A beautiful forest", "A wild forest"
        ]
    }
}

rows = []
image_id = 1

for subject, categories in prompt_data.items():
    base_prompt = categories["Base"][0]
    base_seeds = random.sample(range(10000, 99999), 50)
    for seed in base_seeds:
        rows.append({
            "ID": f"{image_id:03d}",
            "Subject": subject,
            "Evaluation_Type": "Baseline_Accuracy",
            "Prompt": base_prompt,
            "Seed": seed
        })
        image_id += 1

    for prompt in categories["Robustness"]:
        robust_seeds = random.sample(range(10000, 99999), 10)
        for seed in robust_seeds:
            rows.append({
                "ID": f"{image_id:03d}",
                "Subject": subject,
                "Evaluation_Type": "Robustness",
                "Prompt": prompt,
                "Seed": seed
            })
            image_id += 1
    
    for prompt in categories["Bias"]:
        bias_seeds = random.sample(range(10000, 99999), 10)
        for seed in bias_seeds:
            rows.append({
                "ID": f"{image_id:03d}",
                "Subject": subject,
                "Evaluation_Type": "Bias",
                "Prompt": base_prompt,
                "Seed": seed
            })
            image_id += 1

df = pd.DataFrame(rows)
df.to_csv("prompts_english.csv", index=False, encoding="utf-8-sig")