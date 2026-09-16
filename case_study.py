import os
import pandas as pd
from collections import defaultdict


CID2NAME = {
    0:"Model",
    1:"TensorInput",
    2:"Training",
    3:"GPU",
    4:"API",
    5:"Others"
}

CID_LIST = [0,1,2,3,4,5]
def correct_df(df):
    return df[df['True'] == df['pred']]
def mistake_df(df):
    return df[df['True'] != df['pred']]

def main():
    sobert_df = pd.read_csv(os.path.join(exp_root_dir,"exp_random5-3_code/sobert_res/seed_42_1/sobert.csv"))
    codeT5_df = pd.read_csv(os.path.join(exp_root_dir,"exp_random5-3_code/codeT5_res/seed_42_1/codeT5.csv"))
    correct_sobert_df = correct_df(sobert_df)
    mistake_codeT5_df = mistake_df(codeT5_df)
    case_id_dict = defaultdict(set)
    for category_id in CID_LIST:
        df_correct = correct_sobert_df[correct_sobert_df['True'] == category_id]
        df_mistake = mistake_codeT5_df[mistake_codeT5_df['True'] == category_id]
        if df_correct.shape[0] > 0 and df_mistake.shape[0] > 0:
            case_id_dict[category_id] = set(df_correct["Id"]) & set(df_mistake["Id"])
    
    for category_id in CID_LIST:
        print(f"{CID2NAME[category_id]}")
        print(case_id_dict[category_id])

if __name__ == "__main__":
    exp_root_dir = "/data/mml/DL_bug_classification"
    main()