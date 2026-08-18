import os
import json
import re
import time

import anthropic
import openai
import pandas as pd
from openai import OpenAI
from sklearn.model_selection import train_test_split


REQUEST_TIMEOUT = 15 # 单条最大请求时间
MAX_ATTEMPTS = 2 # 单条最多重试步骤
REQUEST_INTERVAL = 3.0 # 请求间隔
RETRY_ROUND_INTERVAL = 10.0  # 一轮失败任务结束后，等待再重试
MAX_RETRY_ROUNDS = 20  # 失败 ID 最多处理10轮，避免无限循环
LABEL_ID_TO_NAME = {
    "0": "Model",
    "1": "Tensors&Inputs",
    "2": "Training",
    "3": "GPU Usage",
    "4": "API",
    "5": "Others",
}


class InvalidModelResponseError(ValueError):
    """模型输出不满足 Answer/Confidence 约束。"""


def build_confidence_query(prompt_template, text):
    """构建自评置信度分类提示。"""
    return prompt_template.replace("<Text>", text).strip()


def normalize_label_id(token):
    if token is None:
        return None

    label_id = str(token).strip()
    if label_id in LABEL_ID_TO_NAME:
        return label_id

    match = re.search(r"(?<!\d)[0-5](?!\d)", label_id)
    if match:
        return match.group(0)
    return None


def build_empty_response_data():
    return {
        "raw_response": "",
        "AnswerId": "",
        "Answer": "",
        "Confidence": "",
    }


def normalize_confidence(confidence):
    if confidence is None or confidence == "":
        return ""
    try:
        confidence_text = str(confidence).strip()
        if confidence_text.endswith("%"):
            return ""
        confidence_value = float(confidence_text)
    except ValueError:
        return ""
    if confidence_value < 0 or confidence_value > 1:
        return ""
    return confidence_value


def parse_answer_and_confidence(response):
    text = response.strip()
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            answer_value = data["Answer"] if "Answer" in data else data.get("answer")
            confidence_value = (
                data["Confidence"] if "Confidence" in data else data.get("confidence")
            )
            answer_id = normalize_label_id(answer_value)
            confidence = normalize_confidence(confidence_value)
            return answer_id or "", confidence
        except json.JSONDecodeError:
            pass

    answer_id = normalize_label_id(text) or ""
    confidence_match = re.search(
        r"(?:confidence|置信度)\s*[:=]\s*([0-9]*\.?[0-9]+%?)",
        text,
        flags=re.IGNORECASE,
    )
    confidence = normalize_confidence(confidence_match.group(1)) if confidence_match else ""
    return answer_id, confidence


def build_response_data(raw_response):
    answer_id, confidence = parse_answer_and_confidence(raw_response)
    if answer_id not in LABEL_ID_TO_NAME:
        raise InvalidModelResponseError(
            f"AnswerId 不是 0-5, raw_response={raw_response!r}"
        )
    if confidence == "":
        raise InvalidModelResponseError(
            f"Confidence 不是 0-1, raw_response={raw_response!r}"
        )

    return {
        "raw_response": raw_response,
        "AnswerId": answer_id,
        "Answer": LABEL_ID_TO_NAME[answer_id],
        "Confidence": confidence,
    }


def create_client(llm_name):
    """根据模型服务创建一个可复用的客户端。"""
    if llm_name == "chatgpt":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("未设置环境变量 OPENAI_API_KEY")

        client_kwargs = {
            "api_key": api_key,
            "timeout": REQUEST_TIMEOUT,
            "max_retries": 0,
        }
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        return OpenAI(**client_kwargs)

    if llm_name == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("未设置环境变量 ANTHROPIC_API_KEY")

        client_kwargs = {
            "api_key": api_key,
            "timeout": REQUEST_TIMEOUT,
            "max_retries": 0,
        }
        base_url = os.getenv("ANTHROPIC_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        return anthropic.Anthropic(**client_kwargs)

    raise ValueError("llm_name 只能是 chatgpt 或 claude")


def query_chatgpt(client:OpenAI, content):
    """使用已创建的 OpenAI client 完成一次独立推理。"""
    gpt_name = "gpt-5.6-sol" # gpt-5.6-sol|gpt-5.5
    print(f"ChatGPT model name:{gpt_name}")
    response = client.chat.completions.create(
        model=gpt_name,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_completion_tokens=100,
    )
    choice = response.choices[0]
    raw_response = choice.message.content or ""
    return build_response_data(raw_response)


def query_claude(client:anthropic.Anthropic, content):
    """使用已创建的 Anthropic client 完成一次独立推理。"""
    claude_name = "claude-opus-4-7" # claude-opus-4-7|claude-opus-4-6
    print(f"Claude model name:{claude_name}")
    response = client.messages.create(
        model=claude_name,
        messages=[{"role": "user", "content": content}],
        max_tokens=100,
        temperature=0,
    )
    raw_response = response.content[0].text or ""
    return build_response_data(raw_response)


def is_retryable_error(error):
    """判断是否为适合重试的瞬时错误。"""
    retryable_errors = (
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.InternalServerError,
        anthropic.APITimeoutError,
        anthropic.APIConnectionError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        InvalidModelResponseError,
    )
    return isinstance(error, retryable_errors)


def query_with_retry(client, query_model, query, sample_id):
    """有限重试单条请求，最终失败时返回错误而不中断任务。"""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempt_start_time = time.monotonic()
        try:
            print(f"开始请求 ID={sample_id},attempt={attempt}/{MAX_ATTEMPTS}",flush=True)
            response = query_model(client, query)
            run_time = time.monotonic() - attempt_start_time
            return response, run_time, None
        except Exception as error:
            run_time = time.monotonic() - attempt_start_time
            error_message = f"{type(error).__name__}: {error}"
            print(
                f"请求失败 ID={sample_id}, attempt={attempt}: "
                f"{error_message}",
                flush=True,
            )

            if attempt >= MAX_ATTEMPTS or not is_retryable_error(error):
                return "", run_time, error_message

            retry_delay = 2 ** (attempt - 1)
            print(f"{retry_delay} 秒后重试 ID={sample_id}", flush=True)
            time.sleep(retry_delay)


def chat(df, llm_name, exp_data_dir, exp_id):
    """推理全部样本，并循环处理失败样本直到全部成功。"""
    client = create_client(llm_name)
    if llm_name == "chatgpt":
        query_model = query_chatgpt
    elif llm_name == "claude":
        query_model = query_claude
    else:
        raise ValueError("llm_name 只能是 chatgpt 或 claude")
    print(f"{llm_name} CSV 将保存模型自评 Confidence。", flush=True)

    # 提示词模板和输出目录只初始化一次。
    with open("prompt_prob.txt", encoding="utf-8") as prompt_file:
        prompt_template = prompt_file.read()

    result_name = f"{llm_name}_prob"
    result_dir = os.path.join(exp_data_dir, f"{result_name}_res", f"repeat_{exp_id}")
    os.makedirs(result_dir,exist_ok=True)
    answer_dir = os.path.join(result_dir, "answer")
    time_dir = os.path.join(result_dir, "time")
    error_dir = os.path.join(result_dir, "error")
    os.makedirs(answer_dir, exist_ok=True)
    os.makedirs(time_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)
    csv_save_path = os.path.join(result_dir, f"{result_name}.csv")

    results_by_id = {}
    pending_rows = list(df.itertuples(index=False))
    retry_round = 1

    try:
        while pending_rows: # 当你还有代办人员
            print(f"开始第 {retry_round} 轮，待处理数量:{len(pending_rows)}",flush=True)
            failed_rows = [] # 准备存储当前轮次还会失败的行
            for row in pending_rows:
                sample_id = row.Id # post id
                query = build_confidence_query(prompt_template, row.Text)
                response_data, run_time, error_message = query_with_retry(
                    client,
                    query_model,
                    query,
                    sample_id,
                )
                status = "success" if error_message is None else "failed"
                error_path = os.path.join(error_dir, f"{sample_id}.txt")

                if status == "success":
                    raw_response = response_data["raw_response"]
                    with open(
                        os.path.join(answer_dir, f"{sample_id}.txt"),
                        "w",
                        encoding="utf-8",
                    ) as answer_file:
                        answer_file.write(raw_response)

                    # error 文件只是临时失败状态，成功后清除。
                    if os.path.exists(error_path):
                        os.remove(error_path)
                else:
                    failed_rows.append(row)
                    with open(
                        error_path,
                        "w",
                        encoding="utf-8",
                    ) as error_file:
                        error_file.write(error_message)
                    response_data = build_empty_response_data()

                with open(
                    os.path.join(time_dir, f"{sample_id}.txt"),
                    "w",
                    encoding="utf-8",
                ) as time_file:
                    time_file.write(str(run_time))

                true_label = getattr(row, "LabelNum", row.Label)
                pred_label = response_data["AnswerId"]
                print(
                    f"{sample_id}\t{status}\t{pred_label}\t"
                    f"{true_label}\t{run_time}",
                    flush=True,
                )
                result_row = {
                    "Id": sample_id,
                    "True": true_label,
                    "pred": pred_label,
                    "Confidence": response_data["Confidence"],
                }
                result_row.update({
                    "Time": run_time,
                    "Status": status,
                    "Error": error_message or "",
                })
                results_by_id[sample_id] = result_row

                # 每完成一条就更新 CSV，避免程序中途退出后丢失进度。
                pd.DataFrame(results_by_id.values()).to_csv(
                    csv_save_path,
                    index=False,
                )
                time.sleep(REQUEST_INTERVAL)

            pending_rows = failed_rows
            if pending_rows:
                if retry_round >= MAX_RETRY_ROUNDS:
                    failed_ids = [row.Id for row in pending_rows]
                    print(
                        f"已达到最大重试轮次 {MAX_RETRY_ROUNDS}，"
                        f"仍失败的 ID: {failed_ids}",
                        flush=True,
                    )
                    break

                print(
                    f"第 {retry_round} 轮仍有 {len(pending_rows)} 个"
                    f"失败 ID，{RETRY_ROUND_INTERVAL} 秒后继续重试",
                    flush=True,
                )
                time.sleep(RETRY_ROUND_INTERVAL)
                retry_round += 1
    finally:
        client.close()

    result_df = pd.DataFrame(results_by_id.values())
    result_df.to_csv(csv_save_path, index=False)
    print(f"结果保存在:{csv_save_path}")
    return result_df

def build_testset(dataset_split_method:str, rs:int) -> pd.DataFrame:
    """
    按照 mmltrain.py 的 dataset_split_method 构建 test set。
    LLM 只对这里返回的 test set 做推理。
    """
    if dataset_split_method == "time":
        return pd.read_csv("reconstruct_dataset/test_dataset.csv").reset_index(drop=True)

    if dataset_split_method == "random":
        df = pd.read_csv("dataset.csv")
        test_size = int(df.shape[0] * 0.15)
        _, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["LabelNum"],
            random_state=int(rs),
        )
        return test_df.reset_index(drop=True)

    raise ValueError("dataset_split_method 只能是 random 或 time")


def build_experiment_configs(experiment_setting:str):
    if experiment_setting == "seed_15":
        return [
            {
                "exp_id": str(repeat),
                "split_seed": 42 + repeat - 1,
                "repeat_id": repeat,
            }
            for repeat in range(1, 1 + 15)
        ]

    if experiment_setting == "seed_5_repeat_3":
        return [
            {
                "exp_id": f"{split_seed}_{repeat_id}",
                "split_seed": split_seed,
                "repeat_id": repeat_id,
            }
            for split_seed in [42, 43, 44, 45, 46]
            for repeat_id in [1, 2, 3]
        ]

    raise ValueError("experiment_setting 只能是 seed_15 或 seed_5_repeat_3")


def main():
    print(f"当前进程 PID: {os.getpid()}")
    start_time = time.monotonic()
    # exp_data_dir = os.path.join(exp_root_dir,"xwj_reproduction")
    exp_data_dir = exp_root_dir
    dataset_split_method = "time"  # random|time
    llm_name = "chatgpt"  # chatgpt|claude
    experiment_setting = "seed_5_repeat_3" # seed_15|seed_5_repeat_3
    experiment_configs = build_experiment_configs(experiment_setting)
    repeat_num = len(experiment_configs)
    print(f"实验设置:{experiment_setting}, 总实验次数:{repeat_num}")
    for experiment_config in experiment_configs:
        exp_id = experiment_config["exp_id"]
        split_seed = experiment_config["split_seed"]
        repeat_id = experiment_config["repeat_id"]
        test_df = build_testset(dataset_split_method, split_seed)
        print(
            f"=== 实验id:{exp_id}, "
            f"重复id:{repeat_id}, "
            f"dataset_split_method:{dataset_split_method}, "
            f"数据集切分随机数种子:{split_seed}, testset大小:{len(test_df)} ==="
        )
        try:
            chat(test_df, llm_name, exp_data_dir, exp_id)
        finally:
            elapsed_seconds = int(time.monotonic() - start_time)
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(
                f"总耗时: {hours:02d}小时 "
                f"{minutes:02d}分钟 {seconds:02d}秒",
                flush=True,
            )

if __name__ == "__main__":
    exp_root_dir = "/data/mml/DL_bug_classification"
    temperature =0.0
    main()
