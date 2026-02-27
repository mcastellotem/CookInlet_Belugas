import json
import os
import requests
import time
import csv

def load_or_create_cache(cache_file):
    name_cache = {}

    if os.path.exists(cache_file):
        with open(cache_file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) == 2:
                    original, standard = row
                    name_cache[original] = standard
    else:
        with open(cache_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["original_name", "standard_name"])

    return name_cache

def save_to_cache(name_cache, cache_file, original, standard):
    """Append a new entry to the CSV and update in-memory cache."""
    # Verify if the original name is already in the cache
    if original in name_cache:
        return
    # Append to the cache
    name_cache[original] = standard
    with open(cache_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([original, standard])

def get_standard_species_name(name, name_cache, cache_file, language='en', max_retries=3, delay=2.0):
    if name in name_cache:
        return name_cache[name]

    url = "https://api.inaturalist.org/v1/search"
    params = {
        'q': name,
        'sources': 'taxa',
        'per_page': 1,
        'locale': language
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    standard_name = data["results"][0]["record"]["name"]
                    print(f"Standardized '{name}' → '{standard_name}'")
                    save_to_cache(name_cache, cache_file, name, standard_name)
                    return standard_name
                else:
                    print(f"No match found for '{name}' (attempt {attempt+1})")

            elif response.status_code == 429:
                wait_time = 30
                print(f"Rate limited for '{name}'. Waiting {wait_time}s before retrying...")
                time.sleep(wait_time)
                continue  # retry same attempt

            else:
                print(f"API error {response.status_code} for '{name}'")

        except Exception as e:
            print(f"Exception with '{name}' (attempt {attempt+1}): {e}")

        time.sleep(delay * (attempt + 1))

    print(f"Fallback: using original name for '{name}'")
    save_to_cache(name_cache, cache_file, name, name)
    return name

def combine_annotation_jsons(json_paths, output_path, cache_file="scientific_species_names_cache.csv"):

    combined_data = {
        "info": {
            "title": "Combined Dataset: ",
            "license": [],
            "publication_date": [],
            "description": [],
            "creators": [],
            "version": [],
            "url": [],
        },
        "categories": [],
        "sounds": [],
        "annotations": []
    }

    category_map = {}  # maps original name -> standardized id
    standard_name_to_id = {}  # maps standardized scientific name -> id
    sound_id_offset = 0
    annotation_id_offset = 0
    next_category_id = 1

    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Merge info fields
        if json_path != json_paths[-1]:
            combined_data["info"]["title"] += data["info"]["title"] + "," + " "
        else:
            combined_data["info"]["title"] += data["info"]["title"]
        combined_data["info"]["license"].append(data["info"]["license"])
        combined_data["info"]["publication_date"].append(data["info"]["publication_date"])
        combined_data["info"]["description"].append(data["info"]["description"])
        combined_data["info"]["creators"].append(data["info"]["creators"])
        combined_data["info"]["version"].append(data["info"]["version"])
        combined_data["info"]["url"].append(data["info"]["url"])

        # Normalize categories using iNaturalist
        name_cache = load_or_create_cache(cache_file)
        local_category_map = {}
        for category in data["categories"]:
            original_name = category["name"]
            standard_name = get_standard_species_name(original_name, name_cache, cache_file)
            if standard_name not in standard_name_to_id:
                standard_name_to_id[standard_name] = next_category_id
                combined_data["categories"].append({
                    "id": next_category_id,
                    "name": standard_name
                })
                next_category_id += 1

            local_category_map[original_name] = standard_name_to_id[standard_name]

        # Merge sounds and update IDs
        sound_id_map = {}
        for sound in data["sounds"]:
            new_sound_id = sound_id_offset
            sound_id_map[sound["id"]] = new_sound_id
            sound["id"] = new_sound_id
            #sound["file_name_path"] = os.path.join(os.path.dirname(json_path), sound["file_name_path"])
            sound["file_name_path"] = sound["file_name_path"]
            combined_data["sounds"].append(sound)
            sound_id_offset += 1

        # Merge annotations with updated IDs
        for annotation in data["annotations"]:
            annotation["anno_id"] += annotation_id_offset
            annotation["sound_id"] = sound_id_map[annotation["sound_id"]]

            original_cat_name = annotation["category"]
            standard_id = local_category_map[original_cat_name]
            annotation["category_id"] = standard_id
            annotation["category"] = [c["name"] for c in combined_data["categories"] if c["id"] == standard_id][0]

            combined_data["annotations"].append(annotation)

        annotation_id_offset += len(data["annotations"])
        time.sleep(1)  # to avoid hitting API rate limits

    # Save merged JSON
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(combined_data, out_file, indent=4)


# List of datasets to combine
json_files = [
    os.path.join("NOAA_Whales", "Humpback_annotations.json"),
    os.path.join("NOAA_Whales", "Orca_annotations.json"),
    os.path.join("NOAA_Whales", "Beluga_annotations.json")
]

output_path = os.path.join("NOAA_Whales", "annotations_combined.json")
combine_annotation_jsons(json_files, output_path)
print(f"✅ Combined dataset json saved in {output_path}")
