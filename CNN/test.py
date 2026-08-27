import pandas as pd
import numpy as np
import plotly.express as px

from archi import Modele

import torch

cols_garde = ["V1.MAG", "V0.MAG", "V2.MAG", "I1.MAG", "I0.MAG", "I2.MAG","FREQ", "DFDT"]



def traite_dataframe(df):

    df = df.copy()
    df["DATE_MESURE"] = pd.to_datetime(df["DATE_MESURE"])
    df.set_index("DATE_MESURE", inplace=True)

    df = df.bfill()


    cols_to_drop = [col for col in df.columns if
                    not col.endswith("MAG") and not col.endswith("FREQ") and not col.endswith("DFDT")]
    df.drop(columns=cols_to_drop,inplace=True)

    for col in df.columns:
        if col.endswith("MAG"):
            df.rename(columns={col: col[-6:]}, inplace=True)
        else:
            df.rename(columns={col: col[-4:]}, inplace=True)


   # start_ts = '2026-07-01 16:49:00'
   # end_ts = '2026-07-01 16:51:00'
   # df = df.loc[start_ts:end_ts]
    return df



def notre_pmu_grapher(df, cols):

    pmu_df = df.copy()
    if cols == "ROCOF":
        cols = next((iter for iter in pmu_df.columns if iter.endswith("DFDT")), None)
        if cols==None:
            return
    if cols.endswith("MAG"):
        car = cols[:2]
        cols = next((iter for iter in pmu_df.columns if iter.endswith(car+".MAG")), None)
        if cols==None:
            return
    if cols.endswith("ANG"):
        car = cols[:2]
        cols = next((iter for iter in pmu_df.columns if iter.endswith(car+".ANG")), None)
        if cols==None:
            return
    if cols == "frequency":
        cols = next((iter for iter in pmu_df.columns if iter.endswith("FREQ")), None)
        if cols==None:
            return
    if len(cols) == 2:
        mag = next((iter for iter in pmu_df.columns if iter.endswith(cols+".MAG")), None)
        ang = next((iter for iter in pmu_df.columns if iter.endswith(cols+".ANG")),None)
        if mag==None or ang==None:
            return
        pmu_df[cols] = pmu_df[mag] * np.cos(np.deg2rad(pmu_df[ang]))

    pmu_df["evenement"] = pmu_df["evenement"] / 4.0

    fig = px.scatter(pmu_df[[cols,"evenement"]],color="evenement")

    fig.show()


def executer(taille_fenetre=4.0):

    chemin = "/home/taklarid/Projects/code/fetch_pmu_data/out_openhistorian/Mesures_OH_PDC_MANTRA_ALBERTVILLE_1__20260715-145500_20260715-165500__50Hz.csv"
    modele_chemin = "/home/taklarid/Projects/code/CNN/best_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Modele().to(device)
    state_dict = torch.load(modele_chemin)
    model.load_state_dict(state_dict)
    model.eval()


    df = pd.read_csv(chemin, sep=";")

    df = traite_dataframe(df)
    df = df[cols_garde]

    df["evenement"] = 0.0

    cur = 0
    while cur + 200 <= df.shape[0]:

        cur_df = df.iloc[cur:cur+200].copy()

        cur_cp = cur_df.copy()
        means = cur_cp[cols_garde].mean()
        stds = cur_cp[cols_garde].std()
        cur_cp[cols_garde] = (cur_cp[cols_garde] - means) / stds

        X = torch.from_numpy(cur_cp[cols_garde].values).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(X)
            probs = torch.sigmoid(logits)
            res = probs.cpu().item()

        col_idx = df.columns.get_loc("evenement")
        df.iloc[cur:cur+200,col_idx] += res
        cur += 50

    cols = ["V1 MAG","frequency"]

    for col in cols:
        notre_pmu_grapher(df, col)



def main():
    executer()


if __name__ == "__main__":
    main()

















