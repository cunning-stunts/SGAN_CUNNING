import os

import math
import pandas as pd
import numpy as np


def get_filename(i):
    return os.path.basename(os.path.normpath(i))


def get_dataframe(ds_location):
    if os.path.exists("df.pkl"):
        print("Loading existing df!")
        return pd.read_pickle("df.pkl")
        # return pickle.load("df.pkl")
    train_df = get_merged_df(ds_location, "train")
    train_df["sirna"] = train_df["sirna"].astype(int)
    train_df["sirna"] = train_df["sirna"].astype(str)
    train_df["well_type"] = train_df["well_type"].replace(np.nan, '', regex=True)
    train_df.to_pickle("df.pkl")
    return train_df


def get_merged_df(ds_location, dataset_type):
    sirna_df = pd.read_csv(os.path.join(ds_location, f"{dataset_type}.csv"))
    controls_df = pd.read_csv(os.path.join(ds_location, f"{dataset_type}_controls.csv"))
    data = []
    tests = [os.path.join(ds_location, dataset_type, t) for t in os.listdir(os.path.join(ds_location, dataset_type))]
    for t in tests:
        print(f"Loading: {str(t)}")
        plates = [os.path.join(t, p) for p in os.listdir(t)]
        for p in plates:
            imgs = [os.path.join(p, i) for i in os.listdir(p)]
            for i in imgs:
                f_name = get_filename(i)
                parts = f_name.split("_")

                well = str(parts[0])
                well_column = well[0]
                well_row = int(well.replace(well_column, ""))
                site = int(str(parts[1]).replace("s", ""))
                microscope_channel = int(str(parts[2]).replace(".png", "").replace("w", ""))
                test = get_filename(t)
                cell_line = test.split("-")[0]
                batch_number = int(test.split("-")[1])
                plate = int(str(get_filename(p)).replace("Plate", ""))

                data.append({
                    "well_column": well_column,
                    "well_row": well_row,
                    "site_num": site,
                    "microscope_channel": microscope_channel,
                    "cell_line": cell_line,
                    "batch_number": batch_number,
                    "plate": plate,
                    "img_location": i,
                    "id_code": f"{cell_line}-{batch_number:02d}_{plate}_{well_column}{well_row:02d}"
                })
    return merge_dfs(sirna_df, pd.DataFrame(data), controls_df)


# SALE NOW ON
def merge_dfs(sirna_df, metadata_df, controls_df):
    metadata_with_sirna = pd.merge(metadata_df, sirna_df[["id_code", "sirna"]], on="id_code", how="left")
    sirnas = metadata_with_sirna.pop("sirna")
    control_merged = pd.merge(
        metadata_with_sirna,
        controls_df[["id_code", "well_type", "sirna"]],
        on="id_code", how="left"
    )
    sirnas2 = control_merged.pop("sirna")
    sirnas3 = []
    for s1, s2 in zip(sirnas, sirnas2):
        if not math.isnan(s1):
            sirnas3.append(s1)
        elif not math.isnan(s2):
            sirnas3.append(s2)
        else:
            raise
    control_merged["sirna"] = sirnas3
    return control_merged


if __name__ == '__main__':
    _train_df, _test_df = get_dataframe("D:\\rxrx1")
