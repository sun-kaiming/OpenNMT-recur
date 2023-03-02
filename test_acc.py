from recur.utils.get_env import get_env
from tqdm import tqdm
import os
import argparse
import csv


def test_acc(src_path,real_path,pre_path):
    with open(real_path,'r',encoding='utf-8')as f1:
        with open(pre_path,'r',encoding='utf-8')as f2:
            with open(src_path,'r',encoding='utf-8')as f3:
                lines1=f1.readlines()
                lines2=f2.readlines()
                lines3=f3.readlines()

                count=0
                for j in range(1):
                    for i in tqdm(range(len(lines1))):
                        if lines1[i].strip()!='EOS' and  lines1[i]==lines2[i] :
                            count+=1
                        else:
                            env = get_env()

                            question = lines3[i].strip().split(' ')
                            real_answer = lines1[i].strip().split(" ")
                            predict_answer = lines2[i].strip().split(" ")
                            error = env.check_prediction(question, real_answer, predict_answer, "tree", 10)
                            if [0.0] * 10 == error:
                                # flag = "预测正确"
                                count += 1

        return count/len(lines1)


def obtain_pred_gongshi(step_nums):
    os.system(f"onmt_translate -model data_recur/model/model_step_{step_nums}.pt -src data_recur/src-test.txt -output data_recur/pred_test_10000.txt -gpu 0 -verbose")
    os.system(f"onmt_translate -model data_recur/model/model_step_{step_nums}.pt -src data_recur/src-val.txt -output data_recur/pred_val_1000.txt -gpu 0 -verbose")
    os.system(f"onmt_translate -model data_recur/model/model_step_{step_nums}.pt -src data_recur/src-oeis.txt -output data_recur/pred_oeis_10000.txt -gpu 0 -verbose")


def save_result(model_name):
    acc_test = test_acc("data_recur/src-test.txt", "data_recur/tgt-test.txt", "data_recur/pred_test_10000.txt")
    acc_val = test_acc("data_recur/src-val.txt", "data_recur/tgt-val.txt", "data_recur/pred_val_1000.txt")
    acc_oeis = test_acc("data_recur/src-oeis.txt", "data_recur/tgt-oeis.txt", "data_recur/pred_oeis_10000.txt")
    print("测试集的acc: ", acc_test)
    print("验证集的acc: ", acc_val)
    print("10000条oeis的acc: ", acc_oeis)

    dir_name = "result"
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    # parser = argparse.ArgumentParser(description="add checkpoints_name")
    # parser.add_argument("--model_name", type=str, default="model_step_i.pt")
    #
    # parsers = parser.parse_args()
    # model_name = parsers.model_name

    with open('result/result.csv', 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(["model_name","val_acc_1000","test_acc_10000"])
        writer.writerow([model_name, acc_val, acc_test,acc_oeis])



if __name__ == '__main__':

    step_nums_list=["260000","270000","280000","290000","300000"]

    for i in range(len(step_nums_list)):
        obtain_pred_gongshi(step_nums_list[i])
        save_result(f"model_step_{step_nums_list[i]}.pt")




