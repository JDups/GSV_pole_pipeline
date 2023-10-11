from utils import get_pole_id_from_filename
from glob import glob
import pandas as pd
import cv2

class img_fetch:

    SUPPORTED_IMAGE_FORMATS = ["png", "jpg"]

    def __init__(self, directory=None):
        self.directory = directory
        
        if self.directory:
            self.add_directory(self.directory)

    def add_directory(self, directory):
            # fl_tp = ["*.png", "*.jpg"]
            fl_tp = [f"*.{f}" for f in self.SUPPORTED_IMAGE_FORMATS]
            pole_pics = []
            for fl in fl_tp:
                pole_pics.extend(glob(directory + fl))

            self.pole_pics_df = pd.DataFrame({'pole_fp': pole_pics})
            self.pole_pics_df['pole_id'] = get_pole_id_from_filename(self.pole_pics_df['pole_fp'])


    def get_batch(self, idn):
        idn = str(idn)
        imgs_fp = [row for row in self.pole_pics_df[self.pole_pics_df["pole_id"]==idn]["pole_fp"]]
        images = [cv2.imread(fp) for fp in imgs_fp]
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        return [{"img": img, "fp": fp.split('\\')[-1]} for img, fp in zip(images, imgs_fp)]