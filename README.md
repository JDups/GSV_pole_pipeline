# GSV-pole-pipeline

## Instalation

The package can be installed with pip.
```
!pip install git+https://git@github.com/JDups/GSV_pole_pipeline.git
```
Or by cloning the repo yourself.

## Minimal example

```python
from GSV_pole_pipeline.loader import GSVFetch
from GSV_pole_pipeline.predictor import YOLOPredictor
from GSV_pole_pipeline.pipeline import Pipeline

csv_path = "my_CSV_path.csv"
GSV_KEY = "my_key"

street_view = GSVFetch(csv_path, GSV_KEY)

weights_path = "YOLOv8-seg_weights.pt"

rules = {
    "interest": ["yolo__pole"],
    "occluding": [
        "yolo__artifact-blur",
        "yolo__artifact-double",
        "yolo__artifact-glare",
    ],
}

yolo_model = YOLOPredictor(weights_path)

pl = Pipeline(street_view, yolo_model, rules)
pl.run(iterations=5)
```

Further documentation coming shortly