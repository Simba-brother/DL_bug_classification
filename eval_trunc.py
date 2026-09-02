import numpy as np
import pandas as pd
import os
def main():
    head_df = pd.read_csv(os.path.join(exp_root_dir,"exp", "sobert_res", "all_res.csv"))
    headTail_df = pd.read_csv(os.path.join(exp_root_dir,"exp", "sobert_res_truncHeadTail", "all_res.csv"))
    col_name_list = head_df.columns.tolist()
    for col_name in col_name_list:
        head_avg = round(np.mean(head_df[col_name].tolist()),4)
        headTail_avg = round(np.mean(headTail_df[col_name].tolist()),4)
        print(f"{col_name}|head_avg:{head_avg}|headTail_avg:{headTail_avg}")
if __name__ == "__main__":
    exp_root_dir = "/data/mml/DL_bug_classification"
    main()