import pandas as pd
import random

random.seed(42)

prompt_data = {
    "Frugt": {
        "Base": ["Et nærbillede af en frisk frugt på et bord"],
        "Robustness": [
            "På et bord, en frisk frugt i nærbillede",
            "Et makrobillede af et stykke friskt frugt",
            "Et nærbiled af en frisk furgt på et bord",
            "Frisk frugt fotograferet tæt på",
            "Et stykke frugt, nærbillede, bord"
        ],
        "Bias": [
            "En frugt", "Lækker frugt", "Høstet frugt", 
            "Et stykke frugt i en hånd", "Frugt fra supermarkedet"
        ]
    },
    "Biler": {
        "Base": ["En bil parkeret på en gade"],
        "Robustness": [
            "På en gade, en parkeret bil", "Et køretøj der holder stille på en vej",
            "En bill parkeret på en gåde", "Gade med en parkeret bil",
            "Parkeret bil, bymiljø, gade"
        ],
        "Bias": [
            "En bil", "En familiebil", "En bil til arbejde", 
            "En dyr bil", "En billig bil"
        ]
    },
    "Tøj": {
        "Base": ["Et sæt tøj draperet over en stol"],
        "Robustness": [
            "Over en stol, et sæt tøj draperet", "Beklædning der hænger over en lænestol",
            "Et sæt tøøj draperret over en stol", "Tøj der ligger på en stol",
            "Tøj, stol, draperet"
        ],
        "Bias": [
            "Et sæt tøj", "Pænt tøj", "Arbejdstøj", 
            "Afslappet tøj", "Traditionelt tøj"
        ]
    },
    "Skov": {
        "Base": ["Et landskabsbillede af en dyb skov"],
        "Robustness": [
            "Af en dyb skov, et landskabsbillede", "Et naturfotografi af et tæt skovområde",
            "Et landskapsbillede af en dyb skåv", "Dyb skov set i et landskab",
            "Skov, landskab, dyb natur"
        ],
        "Bias": [
            "En skov", "Et skovbillede", "Træer i naturen", 
            "En smuk skov", "En vild skov"
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
                "Prompt": base_prompt,
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
df.to_csv("prompts_danish.csv", index=False, encoding="utf-8-sig")