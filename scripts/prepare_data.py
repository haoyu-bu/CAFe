import os
from datasets import load_dataset
from tqdm import tqdm
import json
from pathlib import Path

image_folder = "data/images"

converted_data = []

dataset_list = ['sharegpt4o', 'sharegpt4v(coco)', 'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)', 'image_textualization(filtered)']
print(len(dataset_list))

for dataset in dataset_list:
    
    converted_data = []

    data = load_dataset("LLaVA-OneVision-Data", dataset, split="train")
    print(dataset, len(data))
    
    for da in tqdm(data):
        json_data = {}
        json_data["id"] = da["id"]
        if da["image"] is not None:
            img_path = os.path.join(dataset, da["id"])
            if len(os.path.split(img_path)) > 1:
                parent = os.path.join(image_folder, '/'.join(os.path.split(img_path)[:-1]))
                if not os.path.exists(parent):
                    Path(parent).mkdir(parents=True, exist_ok=True)
            if '.png' == img_path[-4:]:
                json_data["image"] = f"{img_path}"
            else:
                json_data["image"] = f"{img_path}.png"

            da["image"].convert('RGB').save(os.path.join(image_folder, json_data["image"]))
        json_data["conversations"] = da["conversations"]
        converted_data.append(json_data)
    

    with open(os.path.join("data" + dataset + ".json"), "w") as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)

