from cliffs_delta import cliffs_delta



def test():
    ours = [1,2,3,4,5,6,7,8,9,10]
    others = [2,3,4,5,6,7,8,9,10,11]
    delta,info = cliffs_delta(ours, others)
    print()

if __name__ == '__main__':
    test()