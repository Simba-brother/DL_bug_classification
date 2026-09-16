import json
import re
from pathlib import Path
import numpy as np


BASELINE_CLFS = ("LR", "DT", "RF", "SVM", "KNN")
BERT_TRAIN_TIME_PATTERN = re.compile(
    r"总训练耗时：(?P<hours>\d+)小时\s*(?P<minutes>\d+)分钟\s*(?P<seconds>\d+)秒"
)


def time_match_to_seconds(match: re.Match) -> int:
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    return hours * 3600 + minutes * 60 + seconds


def extract_total_train_seconds(log_file_path: str | Path) -> list[int]:
    log_file_path = Path(log_file_path)
    log_text = log_file_path.read_text(encoding="utf-8")
    return [
        time_match_to_seconds(match)
        for match in BERT_TRAIN_TIME_PATTERN.finditer(log_text)
    ]


def get_mean_train_time_for_berts(log_file_path:str):
    total_train_seconds = extract_total_train_seconds(log_file_path)

    if len(total_train_seconds) != 15:
        raise ValueError(
            f"{log_file_path} 中应抽取到15个总训练耗时，"
            f"实际抽取到{len(total_train_seconds)}个"
        )

    # print(f"total_train_seconds:{total_train_seconds}")
    print(f"mean_total_train_seconds:{round(float(np.mean(total_train_seconds)),4)}")




def berts_time_train():
    # Berts 方法的time
    sobert_log_file_path = "logs/random5_3/train_sobert.log"
    codebert_log_file_path = "logs/random5_3/train_codebert.log"
    robert_log_file_path = "logs/random5_3/train_robert.log"
    codeT5_log_file_path = "logs/random5_3/train_codeT5.log"

    get_mean_train_time_for_berts(sobert_log_file_path)
    get_mean_train_time_for_berts(codebert_log_file_path)
    get_mean_train_time_for_berts(robert_log_file_path)
    get_mean_train_time_for_berts(codeT5_log_file_path)

def get_mean_test_time_for_berts(json_file_path,bertname):
    json_file_path = Path(json_file_path)
    with json_file_path.open("r", encoding="utf-8") as file:
        time_data = json.load(file)
    if len(time_data) != 15:
        raise ValueError(
            f"{json_file_path} 中应包含15次实验耗时，"
            f"实际包含{len(time_data)}次"
        )
    time_list = []
    experiment_ids = sorted(time_data.keys(), key=lambda exp_id: int(exp_id))
    for exp_id in experiment_ids:
        time_list.append(time_data[str(exp_id)])
    mean_time = float(np.mean(time_list))
    print(f"{bertname}|test_time_mean:{mean_time}")
    return mean_time

def berts_time_test():
    sobert_json_file_path = "time/sobert_time_test.json"
    robert_json_file_path = "time/robert_time_test.json"
    codebert_json_file_path = "time/codebert_time_test.json"
    codeT5_json_file_path = "time/codeT5_time_test.json"
    get_mean_test_time_for_berts(sobert_json_file_path,"sobert")
    get_mean_test_time_for_berts(robert_json_file_path,"robert")
    get_mean_test_time_for_berts(codebert_json_file_path,"codebert")
    get_mean_test_time_for_berts(codeT5_json_file_path,"codeT5")

def load_baseline_time_json(json_file_path: str | Path) -> dict:
    json_file_path = Path(json_file_path)
    with json_file_path.open("r", encoding="utf-8") as file:
        return json.load(file)
    


def get_mean_train_time_from_json(json_file_path: str | Path, method_name: str) -> dict[str, float]:
    time_data = load_baseline_time_json(json_file_path)

    if len(time_data) != 15:
        raise ValueError(
            f"{json_file_path} 中应包含15次实验耗时，"
            f"实际包含{len(time_data)}次"
        )

    mean_time_dict = {}
    experiment_ids = sorted(time_data.keys(), key=lambda exp_id: int(exp_id))
    for clf_name in BASELINE_CLFS:
        clf_time_list = []
        for experiment_id in experiment_ids:
            experiment_time = time_data[experiment_id]
            if clf_name not in experiment_time:
                raise ValueError(
                    f"{json_file_path} 的第{experiment_id}次实验缺少{clf_name}耗时"
                )
            clf_time_list.append(float(experiment_time[clf_name]))

        if len(clf_time_list) != 15:
            raise ValueError(
                f"{json_file_path} 中 {method_name}-{clf_name} "
                f"应抽取到15个耗时，实际抽取到{len(clf_time_list)}个"
            )

        mean_time = float(np.mean(clf_time_list))
        result_name = f"{method_name}-{clf_name}"
        mean_time_dict[result_name] = mean_time
        print(f"{result_name}:{round(mean_time,4)}")

    return mean_time_dict


def tfidf_word2vec_time():
    # train time
    # tfidf_json_file_path = "time/tfidf_time.json"
    # word2vec_json_file_path = "time/word2vec_time.json"
    
    # test
    tfidf_json_file_path = "time/tfidf_time_test.json"
    word2vec_json_file_path = "time/word2vec_time_test.json"
    

    mean_time_dict = {}
    mean_time_dict.update(
        get_mean_train_time_from_json(tfidf_json_file_path, "TFIDF")
    )
    mean_time_dict.update(
        get_mean_train_time_from_json(word2vec_json_file_path, "Word2Vec")
    )
    return mean_time_dict


def main():
    # berts_time_train()
    tfidf_word2vec_time()
    # berts_time_test()
    
    pass


if __name__ == "__main__":
    main()
