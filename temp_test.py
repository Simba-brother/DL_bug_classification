from cliffs_delta import cliffs_delta
from scipy import stats
import os

def test():
    ours = [1,4,4,4,6,6,7,9,10,10]
    others = [2,3,4,5,6,7,8,9,10,11]
    pvalue = stats.wilcoxon(ours, others).pvalue
    delta,info = cliffs_delta(ours, others)
    print()

def get_csv_files(directory):
    """遍历目录下所有CSV文件并返回文件名列表"""
    csv_files = []
    
    # 遍历目录
    for file in os.listdir(directory):
        # 检查是否为CSV文件
        if file.endswith('.csv'):
            csv_files.append(file)
    
    return csv_files
def test2():
    data = [4,2,5,1]
    data.sort()
    return data
if __name__ == '__main__':
    # test()
    # get_csv_files("/data/mml/DL_bug_classification/remove_word")
    a = test2()
    print(a)