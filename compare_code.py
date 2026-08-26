import os
import pandas as pd

def main():
    withcode_df = pd.read_csv(os.path.join(exp_root_dir,"exp/sobert_res/all_res.csv"))
    nocode_df = pd.read_csv(os.path.join(exp_root_dir,"exp_nocode/sobert_res/all_res.csv"))
    

if __name__ == "__main__":
    exp_root_dir = "/data/mml/DL_bug_classification"
    main()