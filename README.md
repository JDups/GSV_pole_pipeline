# GSV-pole-pipeline

## Instalation

The package can be installed with pip.
```
!pip install git+https://git@github.com/JDups/GSV_pole_pipeline.git
```
Or by cloning the repo yourself.

## Minimal example

The following is a minimal example showing the basic elements required to run the pipeline assuming trained yolo model weights are available.

```python
from GSV_pole_pipeline.loader import GSVFetch
from GSV_pole_pipeline.predictor import YOLOPredictor
from GSV_pole_pipeline.pipeline import Pipeline

csv_path = "my_CSV_path.csv"
GSV_KEY = "my_key"

street_view = GSVFetch(csv_path, GSV_KEY)

weights_path = "YOLOv8-seg_weights.pt"

yolo_model = YOLOPredictor(weights_path)

rules = {
    "interest": ["pole"],
    "occluding": [
        "artifact-blur",
        "artifact-double",
        "artifact-glare",
    ],
}

pl = Pipeline(street_view, yolo_model, rules)
pl.run(iterations=5)
```

## GroundedSAM example

The following is an example showing how to use a text prompted GroundedSAM model in combination with a trined YOLO model.

```python
from GSV_pole_pipeline.loader import GSVFetch
from GSV_pole_pipeline.predictor import CombinedPredictor, GroundedSAMPredictor, YOLOPredictor
from GSV_pole_pipeline.pipeline import Pipeline

csv_path = "my_CSV_path.csv"
GSV_KEY = "my_key"

street_view = GSVFetch(csv_path, GSV_KEY)

yolo_weights_path = "YOLOv8-seg_weights.pt"

yolo_model = YOLOPredictor(yolo_weights_path)

dino_weights_path = "groundingdino_swint_ogc.pth"
pth_to_template = resource_filename(config.__name__, "GroundingDINO_SwinT_OGC.py")
sam_weights_path = "sam_vit_h_4b8939.pth"
sam_enc_version = "vit_h"

class_prompts = {"pole": ["electricity pole"]}

gsam = GroundedSAMPredictor(dino_weights_path, pth_to_template, sam_weights_path, sam_enc_version, class_prompts=class_prompts)


comp = CombinedPredictor([("sam", gsam), ("yolo", yolo_model)])

rules = {
    "interest": ["sam__pole"],
    "occluding": [
        "yolo__artifact-blur",
        "yolo__artifact-double",
        "yolo__artifact-glare",
    ],
}

pl = Pipeline(street_view, comp, rules)
pl.run(iterations=5)
```

Further documentation coming shortly