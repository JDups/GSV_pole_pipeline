class img_fetch:

    def __init__(self, directory=None):
        self.directory = directory
        
        if self.directory:
            pole_pics = glob(directory + "*.png")
            self.pole_pics_df = pd.DataFrame({'pole_fp': pole_pics})
            # print(self.pole_pics_df['pole_fp'].str.split("\\").str[-1])
            self.pole_pics_df['pole_id'] = self.pole_pics_df['pole_fp'].str.split("/").str[-1].str.split("_").str[1].astype(int)

    def get_batch(self, id):
        imgs_fp = [row for row in self.pole_pics_df[self.pole_pics_df["pole_id"]==id]["pole_fp"]]
        images = [cv2.imread(fp) for fp in imgs_fp]
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        return [{"img": img, "fp": fp.split('\\')[-1]} for img, fp in zip(images, imgs_fp)]