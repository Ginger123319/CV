def df_to_coco_json(input_df, coco_json_path, path_col="path", label_col="label"):
    import cv2
    import pandas as pd
    from pathlib import Path
    import json
    assert isinstance(input_df, pd.DataFrame)

    cats = set()
    for _, row in input_df.iterrows():
        annos = json.loads(row["label"])["annotations"]
        for anno in annos:
            cats.add(anno["category_id"])
    cats = sorted(list(cats))

    categories = [{"id": i, "name": str(c)} for i, c in enumerate(cats)]

    images = []
    annotation = []

    anno_id = 0
    for img_id, row in input_df.iterrows():
        annos = json.loads(row["label"])["annotations"]
        if len(annos)==0:
            continue

        img_path = row["path"]
        img_arr = cv2.imread(img_path)
        h, w = img_arr.shape[:2]
        valid = False
        for anno in annos:
            xmin, ymin, xmax, ymax = [float(v) for v in anno["bbox"]]
            xmin = float(max(xmin, 0))
            ymin = float(max(ymin, 0))
            xmax = float(min(xmax, w))
            ymax = float(min(ymax, h))
            if not (xmin < xmax and ymin<ymax):
                continue
            annotation.append({
                "id": anno_id,
                "image_id": img_id,
                "bbox": [xmin, ymin, xmax-xmin, ymax-ymin],
                "category_id": cats.index(anno["category_id"])
            })
            valid = True
        
        if valid:
            images.append({
            "id":img_id,
            "width":w,
            "height":h,
            "file_name":Path(img_path).name
            })
    anno_json = {"images":images,
                "annotations":annotation,
                "categories":categories}
    with open(coco_json_path, "w") as f:
        json.dump(anno_json, f, indent=4)

if __name__=="__main__":
    import pandas as pd
    input_df = pd.read_csv("input_label.csv")
    coco_json_path = "input_label.json"
    path_col = "path"
    label_col = "label"
    df_to_coco_json(input_df=input_df, coco_json_path=coco_json_path)
