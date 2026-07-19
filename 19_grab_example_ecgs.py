"""Pull a few real test-set ECGs spanning the prediction range, for the example figure."""
import pandas as pd, numpy as np
idx = pd.read_parquet("packfinal/index.parquet")
pred = pd.read_parquet("outputs/test_predictions_FINAL.parquet")
W = np.load("packfinal/waves.npy", mmap_mode="r")
print("index:", idx.shape, "| waves:", W.shape)
key = "study_id" if "study_id" in idx.columns else idx.columns[0]
idx_slim = idx.reset_index().rename(columns={"index": "row"})[["row", "study_id"]]
m = idx_slim.merge(
        pred[["study_id", "hfref_le40", "lvef_value", "pred_hfref_le40",
              "as_severe", "pred_as_severe", "age_at_ecg", "sex_male"]],
        on="study_id", how="inner")
print("matched:", len(m))
picks = {}
# a confident true positive, a confident true negative, and a high-risk echo-normal case
tp = m[(m.hfref_le40 == 1)].nlargest(1, "pred_hfref_le40")
tn = m[(m.hfref_le40 == 0)].nsmallest(1, "pred_hfref_le40")
dis = m[(m.lvef_value > 50)].nlargest(1, "pred_hfref_le40")   # echo-normal, model says abnormal
for name, r in [("true_pos", tp), ("true_neg", tn), ("discordant", dis)]:
    if not len(r): continue
    row = r.iloc[0]
    picks[name] = {"row": int(row["row"]), "study_id": int(row["study_id"]),
                   "lvef": float(row["lvef_value"]) if pd.notna(row["lvef_value"]) else None,
                   "hfref": int(row["hfref_le40"]), "pred": float(row["pred_hfref_le40"]),
                   "age": float(row["age_at_ecg"]), "male": int(row["sex_male"])}
np.savez_compressed("example_ecgs.npz",
                    **{k: np.asarray(W[v["row"]], dtype=np.float32) for k, v in picks.items()})
import json; json.dump(picks, open("example_ecgs.json", "w"), indent=1)
print(json.dumps(picks, indent=1))
