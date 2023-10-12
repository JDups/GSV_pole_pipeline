from utils import get_pole_id_from_filename
from glob import glob
import pandas as pd
import cv2


class ImgFetch:
    __supported_image_formats = ["png", "jpg"]

    def __init__(self, directory=None):
        self.directory = directory
        self.dir_div = "/"

        if self.directory:
            self.add_directory(self.directory)

    def add_directory(self, directory):
        if "\\" in self.directory:
            self.dir_div = "\\"
        fl_tp = [f"*.{f}" for f in ImgFetch.__supported_image_formats]

        pole_pics = []
        for fl in fl_tp:
            pole_pics.extend(glob(directory + fl))

        self.pole_pics_df = pd.DataFrame({"img_fp": pole_pics})
        self.pole_pics_df["img_fn"] = (
            self.pole_pics_df["img_fp"].str.split(self.dir_div).str[-1]
        )
        self.pole_pics_df["pole_id"] = self.pole_pics_df["img_fn"].str.split("_").str[1]

    def get_batch(self, idn):
        idn = str(idn)
        imgs_fp = self.pole_pics_df[self.pole_pics_df["pole_id"] == idn][
            "img_fp"
        ].tolist()
        imgs = [cv2.imread(fp) for fp in imgs_fp]
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
        return [
            {"img": img, "fn": fn} for img, fn in zip(imgs, self.pole_pics_df["img_fn"])
        ]
