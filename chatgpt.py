import openai
from openai import OpenAI
import pandas as pd
import time
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
import anthropic
from sklearn.preprocessing import label_binarize

def query_chatgpt(content):
    openai.api_key = 'xxx'
    # openai.base_url = "https://api.chatanywhere.cn"
    # openai.api_key = 'xx'
    messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content}
    ]

    # 调用ChatGPT API
    response = openai.chat.completions.create(
        model="gpt-5.1",  # 或者 "gpt-3.5-turbo" 具体看你要使用的模型
        messages=messages,
        # max_tokens=1500,  # 可选参数，定义响应的最大tokens数
        # n=1,  # 可选参数，定义要生成的响应数量
        # stop=None,  # 可选参数，定义停止生成的标志
        temperature=0,  # 可选参数，定义输出的随机性
    )

    # 输出响应
    # print(response.choices[0].message.content)
    return response.choices[0].message.content

def chatgpt():
    df = pd.read_csv("dataset/dataset.csv")
    res = []
    for i in range(len(df)):
        id = df['Id'][i]
        text = df['Text'][i]
        label = df['Label'][i]
        with open("prompt.txt",encoding='utf-8') as f:
            query = f.read().replace("<Text>",text)
            s_time=time.time()
            response = query_chatgpt(query)
            e_time=time.time()
            run_time = e_time-s_time
            with open(f"chatgpt_res/answer/{id}.txt",'w',encoding='utf-8') as f1:
                f1.write(response)
            with open(f"chatgpt_res/time/{id}.txt",'w',encoding='utf-8') as f2:
                f2.write(str(run_time))            
            print(f"{id}\t{response}\t{label}\t{run_time}")
            res.append({'Id':id, 'Answer':response, 'Label': label, 'Time': run_time})
    res_df = pd.DataFrame(res)
    res_df.to_csv("chatgpt.csv",index=False)

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

def query_claude(content):
    client = anthropic.Anthropic(api_key='xx')
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=0
        
    )
    # print(message.content[0].text)
    return message.content[0].text

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

if __name__ == "__main__":
    # chatgpt()
    # evaluate()
    # query_claude()
    # claude()
    all_time()
    