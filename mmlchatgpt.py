from openai import OpenAI
import pandas as pd
import time
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
import anthropic
from sklearn.preprocessing import label_binarize
import os

def query_chatgpt(content: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    # base_url = os.getenv("OPENAI_BASE_URL")
    base_url = "https://router.latyas.com/v1"

    if not api_key:
        raise RuntimeError("未设置环境变量 OPENAI_API_KEY")

    client = OpenAI(api_key=api_key,base_url=base_url)

    # 结构化message
    messages = [{"role": "user", "content": content}]

    resp = client.chat.completions.create(
        model= 'gpt-5.5',# "gpt-5.6-sol",
        messages=messages,
        temperature=0.2,  # 可选参数，定义输出的随机性
    )
    return resp.choices[0].message.content or ""

def query_claude(content: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    if not api_key:
        raise RuntimeError("未设置环境变量 ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=base_url,
    )
    # 结构化 message
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    resp = client.messages.create(
        model="claude-opus-4-6",
        messages=messages,
        max_tokens=1000,
        temperature=0.2,
    )
    return resp.content[0].text or ""

def chat(df,llm_name):
    '''
    llm_name:chatgpt|claude
    '''
    res = []
    # 遍历测试集
    for i in range(len(df)):
        id = df['Id'][i] # 文本id
        text = df['Text'][i] # 文本内容
        label = df['Label'][i] # 文本标签
        with open("prompt.txt",encoding='utf-8') as f:
            # 读取提示词模版
            query = f.read().replace("<Text>",text) # 把提示词模版中的<Text>替换成文本内容
            s_time=time.time()
            if llm_name == "chatgpt":
                response = query_chatgpt(query)
            elif llm_name == "claude":
                response = query_claude(query)
            else:
                raise Exception("llm_name参数设置错误")
            e_time=time.time()
            run_time = e_time-s_time
            res_save_dir = os.path.join(exp_data_dir,f"{llm_name}_res","answer")
            os.makedirs(res_save_dir,exist_ok=True)
            with open(f"{res_save_dir}/{id}.txt",'w',encoding='utf-8') as f1:
                f1.write(response)
            time_save_dir = os.path.join(exp_data_dir,f"{llm_name}_res","time")
            os.makedirs(time_save_dir,exist_ok=True)
            with open(f"{time_save_dir}/{id}.txt",'w',encoding='utf-8') as f2:
                f2.write(str(run_time))            
            print(f"{id}\t{response}\t{label}\t{run_time}")
            res.append({'Id':id, 'Answer':response, 'Label': label, 'Time': run_time})
    res_df = pd.DataFrame(res)
    csv_save_path = os.path.join(exp_data_dir,f"{llm_name}_res",f"{llm_name}.csv")
    res_df.to_csv(csv_save_path,index=False)
    print(f"结果保存在:{csv_save_path}")

def evaluate():
    df = pd.read_csv("results/claude/claude.csv")
    y_true = df['Label'].tolist()
    y_pred = df['Answer'].tolist()
    label_dict = {'Others': 5, 'API': 4, 'GPU Usage': 3, 'Training': 2, 'Tensors&Inputs': 1, 'Model': 0}
    label_dict2 = {'Others': 5, 'api': 4, 'gpu': 3, 'training': 2, 'tensor': 1, 'model': 0}
    y_true = [label_dict2[label] for label in y_true]
    y_pred = [label_dict[label] for label in y_pred]
    # 计算准确率
    # report = classification_report(y_true, y_pred)
    # print(report)
    acc = {'0':[],'1':[],'2':[],'3':[],'4':[], '5':[], 'all':[]}    # 15次随机结果
    f1 = {'0':[],'1':[],'2':[],'3':[],'4':[], '5':[], 'all':[]}
    auc = {'0':[],'1':[],'2':[],'3':[],'4':[], '5':[], 'all':[]}
    # df = pd.read_csv(f"/data2/xwj/results/sobert_nocode/{rs}_{i}.csv")
    y_true_binary = label_binarize(y_true, classes=range(6))
    y_pred_binary = label_binarize(y_pred, classes=range(6))    
    res = classification_report(y_true, y_pred, output_dict=True)
    acc['all'].append(res['accuracy'])
    f1['all'].append(res['macro avg']['f1-score'])
    auc['all'].append(roc_auc_score(y_true_binary, y_pred_binary, multi_class='ovr'))    
    print(auc)

    for j in range(6):
        auc_value = roc_auc_score(y_true_binary[:, j], y_pred_binary[:, j])
        f1_value = f1_score(y_true_binary[:, j], y_pred_binary[:, j])
        acc_value = accuracy_score(y_true_binary[:, j], y_pred_binary[:, j])
        acc[str(j)].append(auc_value)
        f1[str(j)].append(f1_value)
        auc[str(j)].append(acc_value)   

    data_df = pd.DataFrame()
    for key in auc.keys():
        data_df[f"acc_{key}"] = acc[key]
        data_df[f"f1_{key}"] = f1[key]
        data_df[f"auc_{key}"] = auc[key]

    data_df.to_csv(f"all_res_claude.csv",index=False)

def claude():
    df = pd.read_csv("dataset/dataset.csv")
    res = []
    for i in range(len(df)):
        id = df['Id'][i]
        text = df['Text'][i]
        label = df['Label'][i]
        with open("prompt.txt",encoding='utf-8') as f:
            query = f.read().replace("<Text>",text)
            s_time=time.time()
            response = query_claude(query)
            e_time=time.time()
            run_time = e_time-s_time
            with open(f"results/claude/answer/{id}.txt",'w',encoding='utf-8') as f1:
                f1.write(response)
            with open(f"results/claude/time/{id}.txt",'w',encoding='utf-8') as f2:
                f2.write(str(run_time))            
            print(f"{id}\t{response}\t{label}\t{run_time}")
            res.append({'Id':id, 'Answer':response, 'Label': label, 'Time': run_time})
    res_df = pd.DataFrame(res)
    res_df.to_csv("results/claude/claude.csv",index=False)

def all_time():
    df = pd.read_csv("dataset/dataset.csv")
    claude_time = 0.0
    chatgpt_time = 0.0
    for i in df['Id']:
        with open(f"results/claude/time/{i}.txt",'r',encoding='utf-8') as f:
            run_time = f.read().strip()
            print(run_time)
            claude_time+=float(run_time)
        with open(f"results/chatgpt/time/{i}.txt",'r',encoding='utf-8') as f1:
            run_time = f1.read().strip()
            chatgpt_time+=float(run_time)  

    print(claude_time, chatgpt_time)    
    print(claude_time/600, chatgpt_time/600)     

def main():
    llm_name = "chatgpt" # chatgpt|claude
    chat(test_df,llm_name)

if __name__ == "__main__":
    exp_data_dir = "/data/mml/DL_bug_classification"
    test_df = pd.read_csv("reconstruct_dataset/test_dataset.csv")
    main()
    # evaluate()
    # all_time()