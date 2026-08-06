import os
import time

import anthropic
import openai
import pandas as pd
from openai import OpenAI


REQUEST_TIMEOUT = 30 # 单条最大请求时间
MAX_ATTEMPTS = 2 # 单条最多重试步骤
REQUEST_INTERVAL = 3.0 # 请求间隔5s
RETRY_ROUND_INTERVAL = 10.0  # 一轮失败任务结束后，等待再重试
MAX_RETRY_ROUNDS = 10  # 失败 ID 最多处理10轮，避免无限循环


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
        model="gpt-5.6-sol",
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        # logprobs=True
    )
    return response.choices[0].message.content or ""


def query_claude(client:anthropic.Anthropic, content):
    """使用已创建的 Anthropic client 完成一次独立推理。"""
    response = client.messages.create(
        model="claude-opus-4-6",
        messages=[{"role": "user", "content": content}],
        max_tokens=1000,
        temperature=0.2,
    )
    return response.content[0].text or ""


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
            print(
                f"开始请求 ID={sample_id}, "
                f"attempt={attempt}/{MAX_ATTEMPTS}",
                flush=True,
            )
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

    # 提示词模板和输出目录只初始化一次。
    with open("prompt.txt", encoding="utf-8") as prompt_file:
        prompt_template = prompt_file.read()

    result_dir = os.path.join(exp_data_dir, f"{llm_name}_res", f"repeat_{repeat}")
    os.makedirs(result_dir,exist_ok=True)
    answer_dir = os.path.join(result_dir, "answer")
    time_dir = os.path.join(result_dir, "time")
    error_dir = os.path.join(result_dir, "error")
    os.makedirs(answer_dir, exist_ok=True)
    os.makedirs(time_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)
    csv_save_path = os.path.join(result_dir, f"{llm_name}.csv")

    results_by_id = {}
    pending_rows = list(df.itertuples(index=False))
    retry_round = 1

    try:
        while pending_rows:
            print(
                f"开始第 {retry_round} 轮，待处理数量:"
                f"{len(pending_rows)}",
                flush=True,
            )
            failed_rows = []

            for row in pending_rows:
                sample_id = row.Id
                query = prompt_template.replace("<Text>", row.Text)
                response, run_time, error_message = query_with_retry(
                    client,
                    query_model,
                    query,
                    sample_id,
                )
                status = "success" if error_message is None else "failed"
                error_path = os.path.join(error_dir, f"{sample_id}.txt")

                if status == "success":
                    with open(
                        os.path.join(answer_dir, f"{sample_id}.txt"),
                        "w",
                        encoding="utf-8",
                    ) as answer_file:
                        answer_file.write(response)

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

                with open(
                    os.path.join(time_dir, f"{sample_id}.txt"),
                    "w",
                    encoding="utf-8",
                ) as time_file:
                    time_file.write(str(run_time))

                print(
                    f"{sample_id}\t{status}\t{response}\t"
                    f"{row.Label}\t{run_time}",
                    flush=True,
                )
                results_by_id[sample_id] = {
                    "Id": sample_id,
                    "Answer": response,
                    "Label": row.Label,
                    "Time": run_time,
                    "Status": status,
                    "Error": error_message or "",
                }

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


def main():
    print(f"当前进程 PID: {os.getpid()}")
    repeat_num = 15
    start_time = time.monotonic()
    exp_data_dir = "/data/mml/DL_bug_classification/xw"
    llm_name = "claude"  # chatgpt|claude
    test_df = pd.read_csv("reconstruct_dataset/test_dataset.csv")
    for repeat in range(1,1+repeat_num):
        print(f"=== 实验重复:{repeat} ===")
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
    temperature = 0.0
    main()
