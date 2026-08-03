'''

'''
from __future__ import annotations

import os
import sys
import html
import re
from html.parser import HTMLParser
from pathlib import Path
import time
import pandas as pd


ID_MARKER = b' Id="'
TITLE_MARKER = b' Title="'
BODY_MARKER = b' Body="'


class TextWithoutCodeParser(HTMLParser):
    """提取 HTML 文本，同时忽略整个 code 元素及其内容。"""

    def __init__(self):
        super().__init__(convert_charrefs=True) # convert_charrefs=True 表示自动转义 HTML 实体，例如,&lt; -> <
        self.code_depth = 0 # 当前是否位于<code>标签内部. 0表示不在<code>内部需要保留
        self.text_parts = [] # 用来保存提取到的普通文本

    def handle_starttag(self, tag, attrs):
        '''  每次遇到开始标签时调用，例如：<p>,<li>,<code>等'''
        if tag.lower() == "code":
            self.code_depth += 1 # 如果遇到了<code>深度+1

    def handle_endtag(self, tag):
        '''每次遇到结束标签时调用，例如：</p>,</li>,</code>'''
        if tag.lower() == "code" and self.code_depth > 0:
            # 只要 code_depth > 0，里面的内容就不会保存。
            self.code_depth -= 1

    def handle_data(self, data):
        #   <p>Hello world</p> 保存 Hello world
        if self.code_depth == 0:
            self.text_parts.append(data)

    def get_text(self):
        # 删除换行等空白字符，并将连续空白压缩为一个空格。并且前后删除空格
        return re.sub(r"\s+", " ", " ".join(self.text_parts)).strip()


def extract_attribute(line: bytes, marker: bytes) -> bytes | None:
    """从一行 XML 中提取指定属性的原始字节值。"""
    start = line.find(marker)
    if start == -1:
        return None

    start += len(marker)
    end = line.find(b'"', start)
    if end == -1:
        return None

    return line[start:end]


def extract_plain_text(raw_value: bytes | None) -> str:
    """解码 XML 属性，并删除 HTML 标签和 code 区块。"""
    if raw_value is None:
        return ""

    html_text = html.unescape( # xml转义 -> html
        raw_value.decode("utf-8", errors="replace") # bytes -> str
    )
    parser = TextWithoutCodeParser()
    parser.feed(html_text)
    parser.close()
    return parser.get_text()


def export_post_xml(
    post_ids: list[int],
    posts_xml: Path | str | None = None,
    output_dir: Path | str = "exported_posts",
) -> dict[int, Path]:
    """
    一次扫描 Posts.xml，每个目标 PostId 单独保存一个 XML。

    参数：
        post_ids: 要查找的帖子 ID 列表。
        posts_xml: Posts.xml 路径；不传时使用全局 posts_xml_path。
        output_dir: 每个 Post XML 文件的保存目录。

    返回：
        {post_id: 输出 XML 文件的 Path}。
    """
    remaining = {int(post_id) for post_id in post_ids}
    if not remaining:
        return {}
    target_count = len(remaining)

    source_path = (
        Path(posts_xml)
        if posts_xml is not None
        else Path(posts_xml_path)
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[int, Path] = {}
    file_size = source_path.stat().st_size
    processed_bytes = 0
    report_interval = 1024**3
    next_report = report_interval
    start_time = time.monotonic()

    with source_path.open("rb", buffering=16 * 1024 * 1024) as source_file:
        for line in source_file:
            processed_bytes += len(line)
            raw_id = extract_attribute(line, ID_MARKER)
            if raw_id is not None:
                try:
                    current_post_id = int(raw_id)
                except ValueError:
                    current_post_id = None

                if (
                    current_post_id is not None
                    and current_post_id in remaining
                ):
                    output_path = (
                        output_dir / f"post_{current_post_id}.xml"
                    )
                    with output_path.open("wb") as output_file:
                        output_file.write(
                            b'<?xml version="1.0" encoding="utf-8"?>\n'
                        )
                        output_file.write(b"  ")
                        output_file.write(line.strip())
                        output_file.write(b"\n")
                    saved_paths[current_post_id] = output_path
                    remaining.remove(current_post_id)
                    print(
                        f"PostId={current_post_id} 已保存到："
                        f"{output_path}"
                    )

                    # 全部找到后立即停止，不再扫描剩余文件。
                    if not remaining:
                        break

            if processed_bytes >= next_report:
                percentage = (
                    processed_bytes / file_size * 100
                    if file_size
                    else 0
                )
                elapsed = time.monotonic() - start_time

                print(
                    f"进度：{percentage:.2f}% | "
                    f"已找到：{len(saved_paths)}/{target_count} | "
                    f"已扫描：{processed_bytes / 1024**3:.1f} GiB | "
                    f"耗时：{elapsed:.0f} 秒",
                    file=sys.stderr,
                )

                while next_report <= processed_bytes:
                    next_report += report_interval

    if remaining:
        print(
            f"扫描结束，仍有 {len(remaining)} 个 PostId 未找到："
            f"{sorted(remaining)}",
            file=sys.stderr,
        )

    return saved_paths


def remove_code(
    post_ids: list[int],
    exported_posts_dir: Path | str = "exported_posts",
) -> dict[int, str]:
    """
    请你基于我传给你的post_ids,从exported_posts_dir中的各个xml中，提取Title和Body部分的文本信息
    你需要先移除html转义<code></code>块部分，然后移除所有的html便签，还有移除所有的\n（\n删除后要用空格补）,删除多余的空格。
    最后结果保存为dict,即id:str
    """
    exported_posts_dir = Path(exported_posts_dir)
    post_id_to_text: dict[int, str] = {}
    missing_ids = []

    for post_id_value in post_ids:
        post_id = int(post_id_value)
        post_xml = exported_posts_dir / f"post_{post_id}.xml"

        if not post_xml.is_file():
            missing_ids.append(post_id)
            print(
                f"PostId={post_id} 对应的 XML 文件不存在：{post_xml}",
                file=sys.stderr,
            )
            continue

        post_row = None
        with post_xml.open("rb") as file:
            for line in file:
                raw_id = extract_attribute(line, ID_MARKER)
                if raw_id is None:
                    continue

                try:
                    xml_post_id = int(raw_id)
                except ValueError:
                    continue

                if xml_post_id == post_id:
                    post_row = line
                    break

        if post_row is None:
            missing_ids.append(post_id)
            print(
                f"XML 文件中未找到 PostId={post_id} 的 row："
                f"{post_xml}",
                file=sys.stderr,
            )
            continue

        title = extract_plain_text(
            extract_attribute(post_row, TITLE_MARKER)
        )
        body = extract_plain_text(
            extract_attribute(post_row, BODY_MARKER)
        )

        # Title 可能不存在（例如回答类型的 Post），只连接非空文本。
        text_without_code = " ".join(
            text for text in (title, body) if text
        )
        # 再次归一化空白，确保换行被空格替代且没有多余空格。
        text_without_code = re.sub(
            r"\s+",
            " ",
            text_without_code,
        ).strip()

        post_id_to_text[post_id] = text_without_code
        print(f"PostId={post_id} 文本提取完成")

    if missing_ids:
        print(
            f"共有 {len(missing_ids)} 个 PostId 未成功提取："
            f"{sorted(set(missing_ids))}",
            file=sys.stderr,
        )

    return post_id_to_text

def main():
    posts_xml = Path(posts_xml_path)
    df = pd.read_csv(dataset_csv_path)
    # 数据集中post的ids
    post_ids =  list(df["Id"])
    # remove_code(posts_xml,post_ids)
    output_dir = os.path.join(exp_data_root,"exported_posts")
    os.makedirs(output_dir,exist_ok=True)
    export_post_xml(post_ids, posts_xml,output_dir)

if __name__ == "__main__":
    exp_data_root = "/data/mml/DL_bug_classification"
    dataset_csv_path = "./dataset.csv"
    posts_xml_path = os.path.join(exp_data_root,"Posts.xml")
    main()
