'''
从原贴中抽取时间
'''
import pandas as pd
import os
import sys
import time
import joblib

from pathlib import Path

# 属性前的空格可以避免把 OwnerUserId、ParentId 等误认为 Id
ID_MARKER = b' Id="' # 字节串不是普通字符串(str)
CREATION_DATE_MARKER = b' CreationDate="'

def extract_attribute(line: bytes, marker: bytes) -> bytes | None:
    """
    从一行 XML 中提取指定属性的原始字节值。

    例如：
        line = b'<row Id="123" CreationDate="2020-01-01T00:00:00.000" />'
        marker = b' Id="'

    返回：
        b'123'
    """
    start = line.find(marker)
    if start == -1:
        return None

    start += len(marker)
    end = line.find(b'"', start)

    if end == -1:
        return None

    return line[start:end]

def extract_creation_dates(
    posts_xml: Path,
    post_ids: list[int],
) -> dict[int, str]:
    """
    顺序扫描 Posts.xml，一次查找全部目标 PostId。
    """
    # 待查的post ids
    remaining = set(post_ids)
    # 结果: postId2timestr
    found: dict[int, str] = {}
    # 文件的总大小
    file_size = posts_xml.stat().st_size
    # 处理过的字节数量
    processed_bytes = 0

    # 每扫描 1 GiB 输出一次进度
    report_interval = 1024**3
    next_report = report_interval

    start_time = time.monotonic()

    # 使用较大的文件读取缓冲区（16M），减少磁盘读取调用次数，且文本以二进制模式读取
    with posts_xml.open("rb", buffering=16 * 1024 * 1024) as file: # 
        for line in file:
            # 按行读取，该行的字节数(bytes)
            processed_bytes += len(line)

            raw_id = extract_attribute(line, ID_MARKER)
            if raw_id is not None:
                # 提取到了id
                try:
                    post_id = int(raw_id)
                except ValueError:
                    post_id = None

                if post_id is not None and post_id in remaining:
                    # id在待查的post ids
                    # 提取date
                    raw_creation_date = extract_attribute(
                        line,
                        CREATION_DATE_MARKER,
                    )

                    if raw_creation_date is None:
                        creation_date = ""
                    else:
                        # bytes 转为 str
                        creation_date = raw_creation_date.decode(
                            "ascii",
                            errors="replace", # 如果遇到无法按照 ASCII 解码的字节，不抛出异常，而是用替代字符替代
                        )

                    found[post_id] = creation_date
                    remaining.remove(post_id) # 语出处理过的

                    print(
                        f"找到 PostId={post_id}, "
                        f"CreationDate={creation_date}",
                        file=sys.stderr, # 终端接受err输出信息
                    )

                    # 全部找到后立即停止，不必继续扫描剩余文件
                    if not remaining:
                        break

            if processed_bytes >= next_report:
                # 计算百分比
                percentage = (
                    processed_bytes / file_size * 100
                    if file_size
                    else 0
                )
                elapsed = time.monotonic() - start_time

                print(
                    f"进度：{percentage:.2f}% | "
                    f"已找到：{len(found)}/{len(set(post_ids))} | "
                    f"已扫描：{processed_bytes / 1024**3:.1f} GiB | "
                    f"耗时：{elapsed:.0f} 秒",
                    file=sys.stderr,
                )

                while next_report <= processed_bytes:
                    next_report += report_interval

    return found


def extact_time():
    df = pd.read_csv(dataset_csv_path)
    # 数据集中post的ids
    post_ids =  list(df["Id"])
    # 从Stack Exchange Data Dump下载的post集
    posts_xml = Path(posts_xml_path)
    postId2timestr = extract_creation_dates(posts_xml, post_ids)
    # 结果保存
    joblib.dump(postId2timestr,"postId2timestr.joblib")
    print(f"结果保存在 postId2timestr.joblib")




def add_timeField():
    postId2timestr = joblib.load("postId2timestr.joblib")
    df = pd.read_csv(dataset_csv_path)

    # 按帖子 Id 将创建时间映射到对应的数据行。
    df["createTime"] = df["Id"].map(postId2timestr)

    # 避免部分 Id 未匹配时将缺失的创建时间静默写入 dataset.csv。
    missing_mask = df["createTime"].isna()
    if missing_mask.any():
        missing_ids = df.loc[missing_mask, "Id"].tolist()
        raise ValueError(
            f"有 {len(missing_ids)} 个帖子未找到创建时间，"
            f"未修改 dataset.csv。缺失的 Id：{missing_ids}"
        )

    # index=False 避免把 DataFrame 的行索引写成额外的一列。
    df.to_csv(dataset_csv_path, index=False)
    print(f"已将 createTime 字段写入 {dataset_csv_path}")

def main():
    extact_time()

def read_result():
    # postId2timestr = joblib.load("postId2timestr.joblib")
    print()

if __name__ == "__main__":
    # exp_data_root = "/data/mml/DL_bug_classification"
    dataset_csv_path = "./dataset.csv"
    # posts_xml_path = os.path.join(exp_data_root,"Posts.xml")
    # main()

    # add_timeField()
    # read_result()
