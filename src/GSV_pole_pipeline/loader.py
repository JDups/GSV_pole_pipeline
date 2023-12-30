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
import math
from abc import ABC, abstractmethod
from utils import get_est_heading
import os
import pickle

"""
TODO: Can move up many methods to super calss with some modifications.
      Like setting data_df, IMGFetch will need it's own method to override though
TODO: Load load_gps_csv should just be put into another script that runs when 
      making the gps tracks SO that they get formatted correctly from the begining
TODO: Check that obj_ids are actually present in data_df
"""


class Loader(ABC):
    def __init__(self, csv_file, obj_ids):
        self.log_fp = None
        self.fov = 90
        self.iters = None
        self.data_df = pd.read_csv(csv_file)
        
        if obj_ids:
            self.obj_ids = np.array(obj_ids)
        else:
            self.obj_ids = self.data_df[self.id_col].unique()

    @abstractmethod
    def get_batch(self, idn):
        pass

    def set_log_fp(self, fp):
        self.log_fp = fp

    def output_dict(self, fn=None, img=None, mtdt=None):
        return {
            "fn": fn,
            "img": img,
            "metadata": mtdt,
        }

    def fetch_latlng(self, pid):
        return self.data_df[self.data_df["pole_id"] == pid][
            ["Latitude", "Longitude"]
        ].values[0]

    # Adding these broke ImgFetch
    def __iter__(self):
        self.obj_n = 0
        self.iter_n = 0
        return self

    def __next__(self):
        if self.obj_n == len(self.obj_ids) or self.iter_n == self.iters:
            raise StopIteration

        obj_id = self.obj_ids[self.obj_n]
        batch = self.get_batch(obj_id)

        self.obj_n += 1
        if batch:
            self.iter_n += 1

        return self.obj_n - 1, obj_id, batch


class ImgFetch(Loader):
    __supported_image_formats = ["png", "jpg"]

    def __init__(self, directory):
        super().__init__()
        self.set_directory(directory)

    def set_directory(self, directory):
        self.directory = directory
        fl_tp = [f"*.{f}" for f in ImgFetch.__supported_image_formats]

        pole_imgs = []
        for fl in fl_tp:
            pole_imgs.extend(glob(directory + fl))

        self.data_df = pd.DataFrame({"img_fp": pole_imgs})
        self.data_df["img_fn"] = self.data_df["img_fp"].str.split(os.sep).str[-1]
        self.data_df["pole_id"] = self.data_df["img_fn"].str.split("_").str[1]

    def get_batch(self, idn):
        idn = str(idn)

        imgs = [
            cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
            for fp in self.data_df[self.data_df["pole_id"] == idn]["img_fp"]
        ]

        return [
            self.output_dict(fn=fn, img=img)
            for img, fn in zip(
                imgs, self.data_df[self.data_df["pole_id"] == idn]["img_fn"]
            )
        ]


class GSVFetch(Loader):
    def __init__(self, csv_file, API_key, obj_ids=None, full_360=False):
        self.source = "GSV"
        self.id_col = "pole_id"
        super().__init__(csv_file, obj_ids)
        self.API_key = API_key
        self.full_360 = full_360
        self.saved_queries = {}
        self.saved_imgs = {}
        self.api_defaults = {
            "size": "640x640",
            "fov": "90",
            "pitch": "10",
            "key": self.API_key,
            "source": "outdoor",
        }

    def set_log_fp(self, fp):
        super().set_log_fp(fp)
        self.cache_loc = self.log_fp + "saved_queries" + os.sep
        os.makedirs(self.cache_loc, exist_ok=True)
        try:
            with open(self.cache_loc + "saved_queries.pkl", "rb") as f:
                self.saved_queries = pickle.load(f)
        except:
            self.saved_queries = {}
        try:
            with open(self.cache_loc + "saved_imgs.pkl", "rb") as f:
                self.saved_imgs = pickle.load(f)
        except:
            self.saved_imgs = {}

    def __save_response(self, api_list, api_results):
        if self.log_fp:
            self.saved_queries[str(api_list)] = api_results
            with open(self.cache_loc + "saved_queries.pkl", "wb") as f:
                pickle.dump(self.saved_queries, f)

    def __save_image(self, link, img):
        if self.log_fp:
            self.saved_imgs[link] = img
            with open(self.cache_loc + "saved_imgs.pkl", "wb") as f:
                pickle.dump(self.saved_imgs, f)

    def __log_failed(self, idn, api_results):
        with open(self.log_fp + f"p{idn}_sN.txt", "w") as f:
            print(
                f"\nStreet View request failed. Reponse: {api_results.metadata[0]['status']}\n"
            )
            f.write(api_results.metadata[0]["status"])

    def results_from_loc(self, lat, lng, heading=None):
        apiargs = self.api_defaults.copy()
        apiargs["location"] = f"{lat},{lng}"
        apiargs["heading"] = str(heading)

        api_list = gsv.helpers.api_list(apiargs)

        if str(api_list) in self.saved_queries:
            api_results = self.saved_queries[str(api_list)]
        else:
            api_results = gsv.api.results(api_list)
            self.__save_response(api_list, api_results)

        return api_list, api_results

    def image_from_GSV(self, link):
        if link in self.saved_imgs:
            img = self.saved_imgs[link]
        else:
            img = np.array(Image.open(BytesIO(requests.get(link).content)))
            self.__save_image(link, img)

        return img

    def pic_from_loc(self, idn, lat, lng, heading=None):
        api_list, api_results = self.results_from_loc(lat, lng, heading)
        # print(api_results.metadata)

        if api_results.metadata[0]["status"] != "OK":
            self.__log_failed(idn, api_results)
            return None

        rlat = api_results.metadata[0]["location"]["lat"]  # real Latitude
        rlng = api_results.metadata[0]["location"]["lng"]  # real Longitude
        est_heading = get_est_heading(rlng, rlat, lng, lat)

        if heading is None:
            api_list[0]["heading"] = est_heading

        fn = [
            f"pole_{idn}_heading_{args['heading']}_pitch_{args['pitch']}_zoom_0_fov_{args['fov']}"
            for args in api_list
        ]

        return [
            self.output_dict(
                fn=fn,
                img=self.image_from_GSV(link),
                mtdt=mtdt,
            )
            for link, fn, mtdt in zip(api_results.links, fn, api_results.metadata)
        ]

    def pic_from_loc_360(self, idn, lat, lng):
        return self.pic_from_loc(idn, lat, lng, "0;90;180;270")

    def get_batch(self, idn):
        row = self.data_df[self.data_df["pole_id"] == idn]
        lat, lng = row[["Latitude", "Longitude"]].values[0]
        print(f"Lat: {lat}  Long: {lng}")

        if self.full_360:
            return self.pic_from_loc_360(idn, lat, lng)
        else:
            return self.pic_from_loc(idn, lat, lng)


class DCamFetch(Loader):
    def __init__(self, csv_file, tracks_fp, pics_fp, fov=140, obj_ids=None):
        self.source = "Dashcam"
        self.id_col = "OBJECTID"
        super().__init__(csv_file, obj_ids)
        self.fov = fov
        l_df = []
        for p in tracks_fp:
            print(p)
            l_df.append(self.load_gps_csv(p, pics_fp))
        self.tracks_df = pd.concat(l_df)

    def fetch_latlng(self, pid):
        return self.data_df[self.data_df["OBJECTID"] == pid][["Lat", "Long"]].values[0]

    def load_gps_csv(self, file_path, pics_folder):
        df = pd.read_csv(file_path)

        df = df.drop("Unnamed: 11", axis=1)
        df.columns = df.columns.str.strip()

        df["Point"] = df["Point"] - 1

        df["Bearing_New"] = df["Bearing(Deg)"]

        min_speed = 7
        idx = df.index[df["Speed(mph)"] < min_speed]
        idx = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)

        started_parked = False

        for run in idx:
            if run[0]:
                idx_last = run[0] - 1
                bearing_last = df.iloc[idx_last]["Bearing(Deg)"]
                df.loc[run, "Bearing_New"] = bearing_last
            else:
                started_parked = True
                idx_next = run[-1] + 1
                bearing_next = df.iloc[idx_next]["Bearing(Deg)"]
                df.loc[run, "Bearing_New"] = bearing_next

        if not started_parked:
            df.loc[0, "Bearing_New"] = df.loc[1, "Bearing_New"]

        df["img_path"] = (
            pics_folder
            + df["Filename"].str.split(".").str[0]
            + "\\frame"
            + df["Point"].astype(str)
            + ".jpg"
        )

        return df

    def get_distance(self, x, y, endx, endy):
        dx = endx - x
        dy = endy - y
        return (dx**2 + dy**2) ** (1 / 2)

    def get_est_heading(self, x, y, endx, endy):
        dx = endx - x
        dy = endy - y
        # return (-math.degrees(math.atan2(dy, dx)) + 90) % 360
        return math.degrees(math.atan2(dy, dx)) % 360

    def pic_from_track(self, idn, track, frame_pnt):
        row = self.tracks_df[
            (self.tracks_df["Filename"] == track)
            & (self.tracks_df["Point"] == frame_pnt)
        ]
        img_path = row["img_path"].values[0]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        return [
            self.output_dict(
                fn=f"im_{idn}__{int(row['Bearing_New'].iloc[0])% 360}",
                img=image,
                mtdt={
                    "entry": row,
                    "location": {
                        "lat": row["Latitude(Deg)"].values[0],
                        "lng": row["Longitude(Deg)"].values[0],
                    },
                },
            )
        ]

    def get_batch(self, idn):
        lat, lng = self.fetch_latlng(idn)

        close_tracks = {}

        find_marg = 0.001

        df_close = self.tracks_df[
            (lng - find_marg < self.tracks_df["Longitude(Deg)"])
            & (self.tracks_df["Longitude(Deg)"] < lng + find_marg)
            & (self.tracks_df["Latitude(Deg)"] < lat + find_marg)
            & (lat - find_marg < self.tracks_df["Latitude(Deg)"])
        ]

        for track in df_close["Filename"].unique():
            # print(track)
            track_df = df_close[df_close["Filename"] == track]

            close_tracks[track] = None
            for _, row in track_df.iterrows():
                dist = self.get_distance(
                    row["Longitude(Deg)"], row["Latitude(Deg)"], lng, lat
                )
                if not close_tracks[track]:
                    close_tracks[track] = [row["Point"], dist]
                elif dist < close_tracks[track][1]:
                    close_tracks[track] = [row["Point"], dist]

        print(close_tracks)

        for track in close_tracks:
            found = False
            frame_pnt = close_tracks[track][0]
            while not found:
                row = self.tracks_df[
                    (self.tracks_df["Filename"] == track)
                    & (self.tracks_df["Point"] == frame_pnt)
                ].iloc[0]
                dist = self.get_distance(
                    row["Longitude(Deg)"], row["Latitude(Deg)"], lng, lat
                )
                head = self.get_est_heading(
                    row["Longitude(Deg)"], row["Latitude(Deg)"], lng, lat
                )
                angles = [-row["Bearing_New"] + 90 - 70, -row["Bearing_New"] + 90 + 70]
                if angles[0] < head < angles[1]:
                    found = True
                    break
                if dist > 0.001:
                    break
                if frame_pnt == 0:
                    break
                frame_pnt -= 1

            if found:
                return self.pic_from_track(idn, track, frame_pnt)

        return None
