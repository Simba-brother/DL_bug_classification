from cliffs_delta import cliffs_delta
from scipy import stats

def test():
    ours = [1,4,4,4,6,6,7,9,10,10]
    others = [2,3,4,5,6,7,8,9,10,11]
    pvalue = stats.wilcoxon(ours, others).pvalue
    delta,info = cliffs_delta(ours, others)
    print()

if __name__ == '__main__':
    test()