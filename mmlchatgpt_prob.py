import os
import json
import math
import re
import time

import anthropic
import openai
import pandas as pd
from openai import OpenAI
from sklearn.model_selection import train_test_split


REQUEST_TIMEOUT = 30 # 单条最大请求时间
MAX_ATTEMPTS = 2 # 单条最多重试步骤
REQUEST_INTERVAL = 3.0 # 请求间隔
RETRY_ROUND_INTERVAL = 10.0  # 一轮失败任务结束后，等待再重试
MAX_RETRY_ROUNDS = 10  # 失败 ID 最多处理10轮，避免无限循环
LABEL_ID_TO_NAME = {
    "0": "Model",
    "1": "Tensors&Inputs",
    "2": "Training",
    "3": "GPU Usage",
    "4": "API",
    "5": "Others",
}
PROB_COLUMNS = [f"prob_{label_id}" for label_id in LABEL_ID_TO_NAME]


def build_prob_query(prompt_template, text):
    """构建只输出 0..5 的单 token 分类提示。"""
    return prompt_template.replace("<Text>", text).strip()


def build_claude_confidence_query(prompt_template, text):
    """构建 Claude 自评置信度提示。"""
    query = prompt_template.replace("<Text>", text).strip()
    query = re.sub(r"\n?Answer:\s*$", "", query)
    return f"""{query}

Return exactly one JSON object and no other text.
JSON schema:
{{"Answer":"<one of: 0, 1, 2, 3, 4, 5>","Confidence":<number between 0 and 1>}}
Confidence is your self-estimated probability that Answer is correct.
Answer:"""


def normalize_label_id(token):
    label_id = str(token).strip()
    if label_id in LABEL_ID_TO_NAME:
        return label_id

    match = re.search(r"[0-5]", label_id)
    if match:
        return match.group(0)
    return None


def build_empty_label_probs():
    return {prob_col: "" for prob_col in PROB_COLUMNS}


def build_empty_response_data(llm_name):
    response_data = {
        "raw_response": "",
        "AnswerId": "",
        "Answer": "",
    }
    if llm_name == "chatgpt":
        response_data.update(build_empty_label_probs())
    else:
        response_data["Confidence"] = ""
    return response_data


def normalize_confidence(confidence):
    if confidence is None or confidence == "":
        return ""
    try:
        confidence_text = str(confidence).strip().rstrip("%")
        confidence_value = float(confidence_text)
    except ValueError:
        return ""
    if confidence_value > 1 and confidence_value <= 100:
        confidence_value = confidence_value / 100
    if confidence_value < 0 or confidence_value > 1:
        return ""
    return confidence_value


def parse_claude_answer_and_confidence(response):
    text = response.strip()
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            answer_id = normalize_label_id(data.get("Answer") or data.get("answer"))
            confidence = normalize_confidence(data.get("Confidence") or data.get("confidence"))
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


def extract_label_probs(choice):
    """从首个输出 token 的 top_logprobs 中提取 0..5 的类别概率。"""
    probs = {prob_col: 0.0 for prob_col in PROB_COLUMNS}
    # answer_id in [0-5]
    answer_id = normalize_label_id(choice.message.content or "")

    if not choice.logprobs or not choice.logprobs.content:
        raise RuntimeError("OpenAI response 中没有 logprobs.content")

    first_token_logprob = choice.logprobs.content[0]
    generated_label_id = normalize_label_id(first_token_logprob.token)
    if answer_id is None:
        answer_id = generated_label_id

    top_logprobs = first_token_logprob.top_logprobs or []
    for item in top_logprobs:
        label_id = normalize_label_id(item.token)
        if label_id is None:
            continue
        probs[f"prob_{label_id}"] = max(probs[f"prob_{label_id}"], math.exp(item.logprob))

    if generated_label_id is not None:
        probs[f"prob_{generated_label_id}"] = max(
            probs[f"prob_{generated_label_id}"],
            math.exp(first_token_logprob.logprob),
        )

    prob_sum = sum(probs.values())
    if prob_sum > 0:
        probs = {prob_col: prob_value / prob_sum for prob_col, prob_value in probs.items()}

    if answer_id is None and prob_sum > 0:
        answer_id = max(LABEL_ID_TO_NAME, key=lambda label_id: probs[f"prob_{label_id}"])

    answer = LABEL_ID_TO_NAME.get(answer_id, "")
    return answer_id or "", answer, probs


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
    response = client.chat.completions.create(
        model="gpt-5.6-sol", # gpt-5.6-sol
        messages=[{"role": "user", "content": content}],
        temperature=0,
        logprobs=True,
        top_logprobs=6, # 6个分类
        max_completion_tokens=1,
    )
    choice = response.choices[0]
    print("finish_reason:", choice.finish_reason, flush=True)
    print("content:", repr(choice.message.content), flush=True)
    print("logprobs:", choice.logprobs, flush=True)
    print("usage:", response.usage, flush=True)

    answer_id, answer, probs = extract_label_probs(choice)
    return {
        "raw_response": choice.message.content or "",
        "AnswerId": answer_id,
        "Answer": answer,
        **probs,
    }


def query_claude(client:anthropic.Anthropic, content):
    """使用已创建的 Anthropic client 完成一次独立推理。"""
    response = client.messages.create(
        model="claude-opus-4-6",
        messages=[{"role": "user", "content": content}],
        max_tokens=100,
        temperature=0,
    )
    raw_response = response.content[0].text or ""
    answer_id, confidence = parse_claude_answer_and_confidence(raw_response)
    return {
        "raw_response": raw_response,
        "AnswerId": answer_id,
        "Answer": LABEL_ID_TO_NAME.get(answer_id, ""),
        "Confidence": confidence,
    }


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


def chat(df, llm_name, exp_data_dir,repeat):
    """推理全部样本，并循环处理失败样本直到全部成功。"""
    client = create_client(llm_name)
    query_model = query_chatgpt if llm_name == "chatgpt" else query_claude
    if llm_name == "claude":
        print("Claude Messages API 不返回 logprobs，CSV 将保存模型自评 Confidence。", flush=True)

    # 提示词模板和输出目录只初始化一次。
    with open("prompt_prob.txt", encoding="utf-8") as prompt_file:
        prompt_template = prompt_file.read()

    result_name = f"{llm_name}_prob"
    result_dir = os.path.join(exp_data_dir, f"{result_name}_res", f"repeat_{repeat}")
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
                if llm_name == "chatgpt":
                    query = build_prob_query(prompt_template, row.Text)
                else:
                    query = build_claude_confidence_query(prompt_template, row.Text)
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
                    response_data = build_empty_response_data(llm_name)

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
                }
                if llm_name == "chatgpt":
                    result_row.update({prob_col: response_data[prob_col] for prob_col in PROB_COLUMNS})
                else:
                    result_row["Confidence"] = response_data["Confidence"]
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

def main():
    print(f"当前进程 PID: {os.getpid()}")
    repeat_num = 15
    start_time = time.monotonic()
    exp_data_dir = os.path.join(exp_root_dir,"xwj_reproduction")
    dataset_split_method = "random"  # random|time
    llm_name = "chatgpt"  # chatgpt|claude
    for repeat in range(1,1+repeat_num):
        rs = 42 + repeat - 1
        test_df = build_testset(dataset_split_method, rs)
        print(
            f"=== 实验重复:{repeat}, "
            f"dataset_split_method:{dataset_split_method}, "
            f"数据集切分随机数种子:{rs}, testset大小:{len(test_df)} ==="
        )
        try:
            chat(test_df, llm_name, exp_data_dir,repeat)
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
