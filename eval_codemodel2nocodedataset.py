
import os
import pandas as pd
import numpy as np
from compare import wtl


def main():
    code_df = pd.read_csv(os.path.join(exp_root_dir,"exp", "sobert_res", "all_res.csv"))
    nocode_df = pd.read_csv(os.path.join(exp_root_dir,"exp", "sobert_res_CodeModel2NoCodeDataset", "all_res.csv"))
    col_name_list = code_df.columns.tolist()
    for col_name in col_name_list:
        code_list = code_df[col_name].tolist()
        nocode_list = nocode_df[col_name].tolist()
        h = wtl(code_list,nocode_list)
        code_avg = round(np.mean(code_list),4)
        nocode_avg = round(np.mean(nocode_list),4)

        print(f"{col_name}|code_avg:{code_avg}|nocode_avg:{nocode_avg}|WTL:{h}")

if __name__ == "__main__":
    exp_root_dir = "/data/mml/DL_bug_classification"
    main()