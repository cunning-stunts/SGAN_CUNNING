import os

import math
import pandas as pd
import numpy as np

from consts import DF_LOCATION


def get_filename(i):
    return os.path.basename(os.path.normpath(i))


def get_dataframe(ds_location, is_test=False):
    if is_test:
        folder_name = "test"
    else:
        folder_name = "train"
    filename = f"df_{folder_name}.pkl"
    if os.path.exists(filename):
        print("Loading existing df!")
        return pd.read_pickle(filename)
        # return pickle.load("df.pkl")
    df = get_merged_df(ds_location, folder_name, is_test)
    df = merge_by_channel(df)
    if not is_test:
        df["sirna"] = df["sirna"].astype(int)
    else:
        df.pop("sirna")
    df = df.replace(np.nan, '', regex=True)
    df.to_pickle(filename)
    return df


def merge_by_channel(df):
    df = df.sort_values('img_location')
    img_loc_chan = pd.DataFrame(df.pop('microscope_channel'))
    img_loc_chan['img_loc'] = df.pop('img_location')
    df = df.drop_duplicates()
    channels = img_loc_chan['microscope_channel'].drop_duplicates()
    for i in channels:
        channel_n = img_loc_chan.loc[img_loc_chan['microscope_channel'].isin([i])]
        channel_n.rename(columns={'img_loc': f'img_loc_{i}'}, inplace=True)
        df = df.reset_index(drop=True)
        channel_n = channel_n.reset_index(drop=True)
        df = pd.concat([df, channel_n[f'img_loc_{i}']], axis=1)
    return df


def get_merged_df(ds_location, dataset_type, is_test):
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

    return add_sirna(sirna_df, pd.DataFrame(data), controls_df, is_test)


def add_sirna(sirna_df, metadata_df, controls_df, is_test):
    if is_test:
        metadata_with_sirna = pd.merge(metadata_df, sirna_df[["id_code"]], on="id_code", how="left")
    else:
        metadata_with_sirna = pd.merge(metadata_df, sirna_df[["id_code", "sirna"]], on="id_code", how="left")
        sirnas = metadata_with_sirna.pop("sirna")
    control_merged = pd.merge(
        metadata_with_sirna,
        controls_df[["id_code", "well_type", "sirna"]],
        on="id_code", how="left"
    )
    if not is_test:
        set_sirna(control_merged, sirnas)
    return control_merged


def set_sirna(control_merged, sirnas):
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


if __name__ == '__main__':
    test_df = get_dataframe(DF_LOCATION, is_test=True)
    train_df = get_dataframe(DF_LOCATION, is_test=False)
    print("")
