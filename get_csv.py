import os
import pandas as pd

folder = "Face_detection_df-/processed_data"
lis = []
for img_path in os.listdir(folder):
    if img_path.endswith("without_mask.png"):
        lis.append([img_path, 0])
    if img_path.endswith("with_mask.png"):
        lis.append([img_path, 1])

df = pd.DataFrame(lis)
df.to_csv("data_class.csv")
