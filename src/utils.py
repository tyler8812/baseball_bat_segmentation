
# get baseball bat dictionary in label json file
def get_baseball_bat_dicts(img_dir):
    import os
    import json
    import cv2
    import numpy as np
    from detectron2.structures import BoxMode
    json_file = os.path.join("../" + img_dir, "via_region.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        filename = os.path.join("../" + img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for i in range(len(annos)):
            anno = annos[i]
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register():
    from detectron2.data import MetadataCatalog, DatasetCatalog
    for d in ["train", "test"]:
        DatasetCatalog.register(
            "baseball_bat_" + d, lambda d=d: get_baseball_bat_dicts("baseball_bat/" + d)
        )
        MetadataCatalog.get("baseball_bat_" + d).set(thing_classes=["baseball_bat"])
    baseball_bat_metadata = MetadataCatalog.get("baseball_bat_train")
    return baseball_bat_metadata
