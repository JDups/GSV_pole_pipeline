# from utils import get_pole_id_from_filename
from glob import glob
import pandas as pd
import cv2
import google_streetview as gsv
import google_streetview.helpers
import google_streetview.api
import requests
from PIL import Image
from io import BytesIO
import numpy as np


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

        pole_imgs = []
        for fl in fl_tp:
            pole_imgs.extend(glob(directory + fl))

        self.data_df = pd.DataFrame({"img_fp": pole_imgs})
        self.data_df["img_fn"] = self.data_df["img_fp"].str.split(self.dir_div).str[-1]
        self.data_df["pole_id"] = self.data_df["img_fn"].str.split("_").str[1]

    def get_batch(self, idn):
        idn = str(idn)

        imgs = [
            cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
            for fp in self.data_df[self.data_df["pole_id"] == idn]["img_fp"]
        ]

        return [
            {"img": img, "fn": fn}
            for img, fn in zip(
                imgs, self.data_df[self.data_df["pole_id"] == idn]["img_fn"]
            )
        ]


class GSVFetch:
    def __init__(self, csv_file, API_key):
        self.data_df = pd.read_csv(csv_file)
        self.API_key = API_key

    def get_batch(self, pid):
        row = self.data_df[self.data_df["pole_id"] == pid]
        lat = row["Latitude"].values[0]
        lng = row["Longitude"].values[0]
        print(f"Lat: {lat}  Long: {lng}")

        apiargs = {
            "location": f"{lat},{lng}",
            "size": "640x640",
            "heading": "0;90;180;270",
            "fov": "90",
            "pitch": "10",
            "key": self.API_key,
        }
        api_list = gsv.helpers.api_list(apiargs)
        api_results = gsv.api.results(api_list)
        api_results.save_links("links.txt")
        api_results.save_metadata("metadata.json")

        # return api_results.metadata
        # pole_1114_heading_0_pitch_10_zoom_0_fov_90

        fn = [
            f"pole_{pid}_heading_{args['heading']}_pitch_{args['pitch']}_zoom_0_fov_{args['fov']}"
            for args in api_list
        ]

        return [
            {
                "img": np.array(Image.open(BytesIO(requests.get(link).content))),
                "fn": fn,
                "metadata": mtdt,
            }
            for link, fn, mtdt in zip(api_results.links, fn, api_results.metadata)
        ]

    """
    I think implementing a method to get an image set from lat and long will let me 
    call it in the pieline for readjusting and that method can also be called in get_batch

    An arg to either get 360 or two call process where we get picture with heading based on
    difference of coordinates. Either way should work with pipeline with no changes.
    """
