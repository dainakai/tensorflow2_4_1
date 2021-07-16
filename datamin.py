import glob
import numpy as np

def datamining(num):
    data = []
    path = f"./saveddata/n_{num}/*"
    files = glob.glob(path)
    for item in files:
        f = open(item,"r")
        f.readline()
        text = f.readline()
        tmp = float(text.replace("test data accyracy : ","").replace("\n",""))
        if num == 1 or num == 2 or num == 3:
            met = 0.8
        elif num == 4 or num == 5 or num == 6:
            met = 0.7
        elif num == 7 or num == 8 or num == 9:
            met = 0.6
        elif num == 10:
            met = 0.55
        else:
            met = 0.5
        if tmp <= met:
            data.append([int(num),tmp])

    # np_tmp = np.array(data)
    # print(len(data))
    # avr = np_tmp.sum(axis=0)[1]/len(data)
    # print(f"ave : {avr}")

    # return data, avr
    return data

def main():
    full_array = []
    # avr_array = []
    for i in range(1,12):
        # data, avr = datamining(i)
        data = datamining(i)
        full_array.extend(data)
        # avr_array.append(avr)
    print(full_array)
    # print(avr_array)
    array = np.array(full_array)
    np.savetxt("./result/test3.txt",array)

    # result_avr = np.zeros([11,2])
    # for i in range(11):
    #     result_avr[i,0] = i+1
    #     result_avr[i,1] = avr_array[i]
    # print(result_avr)

    # np.savetxt("./result/avr2.txt",result_avr)


if __name__ == "__main__":
    main()

