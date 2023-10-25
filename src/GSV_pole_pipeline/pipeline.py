class Pipeline:
    def __init__(self, loader=None, predictors=[]):
        self.loader = loader


"""
    General idea will be
        - Loader object that returns batch of images that are to be processed at a time.
          Loader could also possibly be the one determine the end condition of the whole 
          process, with it keeping track internally. With a Pipeline level override.
        
        - Process objects that will return masks (or possibly other types of predictions)
        
        - Rules/Decison object that will take the the previous steps predictions and make decision.
          The Pipeline could be given a list of what prediction represent what we want and which represent what we don't want.
          It could then use that list to cross compare the outputs of multiple process objects

Some test code chucked in here for now
    rules = {
        "interest": ["mock__pole-hydro"],
        "occluding": ["margin__margin"],
    }

    for pid in ld.pole_imgs_df["pole_id"].unique():
        print(f"\nPole ID: {pid}")
        batch = ld.get_batch(pid)
        # preds = fake_model.predict(batch)
        # preds = margin_pred.predict(batch)
        preds = comp.predict(batch)

        # print(preds[0])

        # show_masks_indiv(preds, rules)
        # show_masks_comb(preds, rules)

        largest = {
            "fn": None,
            "interest": None,
            "occluding": None,
            "orig_img": None,
        }

        for p in preds:
            print(f"File: {p['fn']}")
            occl = np.zeros(p["orig_img"].shape[:2], dtype=bool)

            for clss, m in zip(p["out"]["class"], p["out"]["mask"]):
                if clss in rules["occluding"]:
                    occl = np.logical_or(occl, m)

                if clss in rules["interest"]:
                    if largest["fn"] is None or m.sum() > largest["interest"].sum():
                        largest = {
                            "fn": p["fn"],
                            "interest": m,
                            "occluding": None,
                            "orig_img": p["orig_img"],
                        }

            if largest["fn"]:
                # print(largest["interest"].sum())
                largest["occluding"] = occl
            else:
                print("No pole yet")

        if largest["fn"]:
            print(np.logical_and(largest["interest"], largest["occluding"]).sum())
            show_mask(
                largest["orig_img"],
                p_msk=largest["interest"],
                n_msk=largest["occluding"],
            )

        # break          
"""
