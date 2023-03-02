import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import uuid
import cv2
from tqdm import tqdm


RAW_DATA_PATH = Path("data/raw/CHIL")
XLSX_PATH = RAW_DATA_PATH / "CHIL_uwin_mange_Marit_07242020.xlsx"
COCO_PATH = Path("data/processed/CHIL/CHIL_uwin_mange_Marit_07242020.json")
CONTRIBUTOR = "Maureen Murray"
EXTENDED_DESCRIPTION = """updateCommonName	Confirm species ID
updateNumIndividuals	Confirm number of individuals in photo
File_name_order	consecutive numbering
Inspected	1 indicates that this photo was checked
Mange_signs_present	1 indicates that photo contains at least one coyote exhibiting signs of mange including hair loss and/or lesions on the tail, legs, body, or face
In_color	1 indicates that the photo was in color, which likely improves the detectability of mange
Season	Calendar season in which the photo was taken (Spring, Summer, Fall, Winter)
Year	Year in which the photo was taken
propbodyvis	Proportion of the coyote's body that is visible in the photo, expressed from 0 - 1. For example, if a coyote is in perfect profile from nose to tail it would get a value of 0.5
propbodyvismange	Proportion of the coyote's body that is visible and affected by mange
severity	Mild < 25%, Moderate = 25 - 50%, Severe > 50%
confidence	High, medium, low - subjective!
flagged for follow up	1 indicates that coyote exhibits potential signs of mange but confidence is low because of angle, photo quality, ambiguous signs, etc."""


def main():
    df = pd.read_excel(XLSX_PATH)
    try:
        with open(COCO_PATH, "r") as f:
            coco = json.load(f)
            version = int(coco["info"]["version"]) + 1
    except FileNotFoundError:
        version = 1
    info = {
        "version": str(version),
        "year": 2020,
        "description": "Chicago Urban Urban Wildlife Information Network Mange Dataset\n"
        + EXTENDED_DESCRIPTION,
        "contributor": CONTRIBUTOR,
        "date_created": datetime.today().date().isoformat(),
    }
    images = []
    categories = [
        {"id": 1, "name": "coyote", "supercategory": "mange"},
        {
            "id": 2,
            "name": "coyote",
            "supercategory": "no_mange",
        },
        {"id": 3, "name": "coyote", "supercategory": "unknown"},
        {"id": 4, "name": "fox", "supercategory": "mange"},
        {"id": 5, "name": "fox", "supercategory": "no_mange"},
        {"id": 6, "name": "fox", "supercategory": "unknown"},
    ]
    annotations = []
    sequence = []
    seq_id = None
    seq_mange = None
    prev_vid_number = None
    prev_frame_number = None
    for _, row in tqdm(df.iterrows(), total=len(df)):
        corrupt = False
        width = None
        height = None
        photo_path = RAW_DATA_PATH / row["photoName"]
        # split file io into parallelizable loop
        try:
            img = cv2.imread(str(photo_path))
            width = img.shape[1]
            height = img.shape[0]
        except Exception as e:
            print(e)
            corrupt = True
        # NOTE: can only approximate sequences using frame numbers
        # TODO: try ocr
        vid_number, frame_number = row["photoName"].split("-")
        vid_number = int(vid_number.replace("VID", ""))
        frame_number = int(frame_number.replace(".jpg", ""))
        if (
            frame_number - 1 != prev_frame_number
            or vid_number != prev_vid_number
            or seq_mange != row["Mange_signs_present"]
        ):
            seq_id = str(uuid.uuid1())
            for frame_num, image in enumerate(sequence):
                image["seq_id"] = seq_id
                image["seq_num_frames"] = len(sequence)
                image["frame_num"] = frame_num
                images.append(image)
            sequence = []
        prev_frame_number = frame_number
        prev_vid_number = vid_number
        seq_mange = row["Mange_signs_present"]

        image = {
            "id": str(uuid.uuid1()),
            "file_name": row["photoName"],
            "width": width,
            "height": height,
            "rights_holder": CONTRIBUTOR,
            "datetime": None,
            "location": row["locationName"],
            "corrupt": corrupt,
        }
        sequence.append(image)

        # annotations
        commonName = (
            row["updateCommonName"]
            if not pd.isna(row["updateCommonName"])
            else row["commonName"]
        )
        if commonName == "Coyote":
            if row["Mange_signs_present"] == 1:
                category_id = 1
            elif row["Mange_signs_present"] == 0:
                category_id = 2
            elif pd.isna(row["Mange_signs_present"]):
                category_id = 3
            else:
                raise Exception(
                    f"Unknown Mange_signs_present: {row['Mange_signs_present']}"
                )
        elif commonName == "Red fox":
            if row["Mange_signs_present"] == 1:
                category_id = 4
            elif row["Mange_signs_present"] == 0:
                category_id = 5
            elif pd.isna(row["Mange_signs_present"]):
                category_id = 6
            else:
                raise Exception(
                    f"Unknown Mange_signs_present: {row['Mange_signs_present']}"
                )
        else:
            print(row)
            raise Exception(f"Unknown commonName: {commonName}")

        numIndividuals = (
            int(row["updateNumIndividuals"])
            if not pd.isna(row["updateNumIndividuals"])
            else int(row["numIndividuals"])
        )

        # some notes are misplaced in the good picture quality column
        good_picture_quality = None
        try:
            good_picture_quality = (
                int(row["good picture quality"])
                if not pd.isna(row["good picture quality"])
                else None
            )
        except ValueError:
            if pd.isna(row["Notes"]):
                row["Notes"] = row["good picture quality"]

        annotation = {
            "id": str(uuid.uuid1()),
            "image_id": image["id"],
            "category_id": category_id,
            "bbox": [0, 0, width, height - 198],
            "sequence_level_annotation": False,
            "city": row["City"] if not pd.isna(row["City"]) else None,
            "inspected": int(row["Inspected"])
            if not pd.isna(row["Inspected"])
            else 0,
            "num_individuals": numIndividuals,
            "in_color": int(row["In_color"])
            if not pd.isna(row["In_color"])
            else None,
            "season": row["Season"] if not pd.isna(row["Season"]) else None,
            "year": int(row["Year"]) if not pd.isna(row["Year"]) else None,
            "propbodyvis": row["propbodyvis"]
            if not pd.isna(row["propbodyvis"])
            else None,
            "propbodyvismange": row["propbodyvismange"]
            if not pd.isna(row["propbodyvismange"])
            else None,
            "severity": row["severity"]
            if not pd.isna(row["severity"])
            else None,
            "confidence": row["confidence"]
            if not pd.isna(row["confidence"])
            else None,
            "flagged_for_follow_up": int(row["flagged for follow up"])
            if not pd.isna(row["flagged for follow up"])
            else None,
            "good_picture_quality": good_picture_quality,
            "notes": row["Notes"] if not pd.isna(row["Notes"]) else None,
        }
        annotations.append(annotation)

    # flush last sequence
    seq_id = str(uuid.uuid1())
    for frame_num, image in enumerate(sequence):
        image["seq_id"] = seq_id
        image["seq_num_frames"] = len(sequence)
        image["frame_num"] = frame_num
        images.append(image)

    assert len(images) == len(df)
    assert len(annotations) == len(df)

    coco = {
        "info": info,
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }
    with open(COCO_PATH, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=4)


if __name__ == "__main__":
    main()
