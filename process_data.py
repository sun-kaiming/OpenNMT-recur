import json


def obtain_data(sta_id, end_id, path, name):
    with open('1000M/1000M.prefix', 'r', encoding='utf-8') as f:
        with open(f'{path}/{name}.txt', 'w', encoding='utf-8') as f2:
            with open('500M_data/data.prefix', 'r', encoding='utf-8') as f3:
                lines1 = f.readlines()
                lines2 = f3.readlines()
                print(len(lines1))
                print(len(lines2))
                print("#" * 50)

                endid2=0
                if end_id>len(lines1):
                    endid2=len(lines1)
                count=0
                for i in range(0, 7913850):
                    f2.write(lines1[i])
                    # if i>7913850:
                    #     print(lines1[i])
                    count+=1
                for i in range(sta_id, end_id-endid2+2):
                    f2.write(lines2[i])
                    count+=1
                print("写入的总行数",count)



def obtain_valid_test_data(sta_id, end_id, path, name):
    with open('data.prefix', 'r', encoding='utf-8') as f:
        with open(f'{path}/{name}.txt', 'w', encoding='utf-8') as f2:
            lines = f.readlines()
            print(len(lines))
            print("#" * 50)
            for i in range(sta_id, end_id):
                f2.write(lines[i])


def process_oeis_data(num, path,len2):  # 处理oeis数据成为该代码能够处理的形式
    with open('OEIS/stripped', 'r', encoding='utf-8') as f:
        with open(f'{path}/oeis_{num}_{len2}.json', 'w', encoding='utf-8') as f2:
            lines = f.readlines()
            oeis_dict = {
                "name": "",
                "x1": "",
                "x2": "EOS",
                "tree": "",
                "n_input_points": "30",
                "n_ops": "1",
                "n_recurrence_degree": "0"
            }
            # print(str(oeis_dict).replace('\'','"'))
            # exit()
            a=len(lines)
            if num>len(lines):
                num=len(lines)
            elif (num+4)>len(lines):
                num=len(lines)
            else:
                num+=4

            count=0
            for i in range(4, num):
                # print(lines[i])
                if "A" in lines[i]:
                    oeis_dict['name'] = lines[i][:7]
                    temp=process_oeis_input_seq(lines[i],len2)
                    if temp!='false':


                        oeis_dict['x1'] = temp
                        f2.write(str(oeis_dict).replace('\'','"')+'\n')
                        count += 1
                        if count>=10000:
                            break


def process_oeis_input_seq(oeis_seq,num2):  # 处理oeis整数序列： 例如 "1,-1,1,-2,2,-4567893"=> "+ 1 - 1 1 - 2 2 - 456 7893"
    # oeis_seq='A000087 ,2,1,2,4,10,37,138,628,2972,14903,76994,409594,-2222628,12281570,-68864086,391120036,2246122574,13025720000,76101450042,449105860008,2666126033850,15925105028685,95664343622234,577651490729530,'

    oeis_seq=oeis_seq[9:-2].strip().split(',')
    count=0
    # if len(oeis_seq) <= 10:
    #     print(len(oeis_seq))
        # print(oeis_seq)
        # print("$$$$$$$")
    if len(oeis_seq)>=num2:

        oeis_seq=oeis_seq[:num2]
    else:
        return "false"
    # print(oeis_seq)
    # print(len(oeis_seq))
    # exit()
    oeis_seq2=[]
    for num in oeis_seq:
        if len(num)>4:
            num=process_big_num(num)

        if '-' not in num:
            oeis_seq2.append('+')
            oeis_seq2.extend(num.split())
        else:
            oeis_seq2.append('-')
            oeis_seq2.extend(num[1:].split())
    # print(oeis_seq)
    # print(type(oeis_seq[0]))

    # print(oeis_seq2)
    oeis_seq3=[]
    for n in oeis_seq2:
        if len(n)==4:
            if n == "0000":
                oeis_seq3.append('0')
            elif n[:3] == "000":
                oeis_seq3.append(n[-1])
            elif n[:2] == "00":
                oeis_seq3.append(n[2:])
            elif n[:1] == "0":
                oeis_seq3.append(n[1:])
            else:
                oeis_seq3.append(n)
        else:
            oeis_seq3.append(n)
    # print(oeis_seq3)
    # exit()
    return ' '.join(oeis_seq3)

def process_big_num(big_num):#处理大数，把大数分割为小于1000的数，例如：12345321=> 1234 5321
    count=0
    big_num_len=len(big_num)
    big_num_split=big_num
    inster_index=[]# 在大数中插入空格下标
    for i in range(1,big_num_len):
        j=4*i
        k=big_num_len-j
        if k<0:
            break
        inster_index.append(k)
    # print(inster_index)
    for id in inster_index:
        big_num_split=str_insert(big_num_split,id,' ')
    big_num_split=big_num_split.strip()


    # print("bignum:",big_num_split)
    return big_num_split


# 在字符串指定位置插入字符
# str_origin：源字符串  pos：插入位置  str_add：待插入的字符串
#
def str_insert(str_origin, pos, str_add):
    str_list = list(str_origin)    # 字符串转list
    str_list.insert(pos, str_add)  # 在指定位置插入字符串
    str_out = ''.join(str_list)    # 空字符连接
    return  str_out

def delete_same_data():
    with open('test/10000.txt','r',encoding='utf-8') as f:
        with open('test/10000_delete_same_term.txt','w',encoding='utf-8') as f2:
            lines=f.readlines()
            save_list=[]
            for line in lines:
                json_dump=json.loads(line)
                # if json_dump['x1'][:39]:
                # print(json_dump['x1'].split())
                qian_n_term=" ".join(json_dump['x1'].split()[:20])
                # print(" ".join(json_dump['x1'].split()[:20]))
                flag=0 #判断数列前10项有没有在save_list中出现
                for lis in save_list:
                    if qian_n_term in lis['x1']:
                        print(lis)
                        flag=1
                if flag==0:
                    save_list.append(json_dump)
            print(len(save_list))
            for lis in save_list:
                f2.write(str(lis).replace('\'','"')+'\n')

def decode(lst):

    if len(lst) == 0:
        return None
    res = []
    if lst[0] in ["+", "-"]:
        curr_group = [lst[0]]
    else:
        return None
    if lst[-1] in ["+", "-"]:
        return None

    for x in lst[1:]:
        if x in ["+", "-"]:
            if len(curr_group) > 1:
                sign = 1 if curr_group[0] == "+" else -1
                value = 0
                for elem in curr_group[1:]:
                    value = value * 10000 + int(elem)
                res.append(sign * value)
                curr_group = [x]
            else:
                return None
        else:
            curr_group.append(x)
    if len(curr_group) > 1:
        sign = 1 if curr_group[0] == "+" else -1
        value = 0
        for elem in curr_group[1:]:
            value = value * 10000 + int(elem)
        res.append(sign * value)
    return res

def trans_test():
    with open('test/oeis_10000.json','r',encoding='utf-8')as f1:
        with open('test/only_data_oeis10000.txt','w',encoding='utf-8')as f2:
            lines=f1.readlines()

            for line in lines:
                line=json.loads(line)
                x1=line['x1']
                res_x1=decode(x1.split(' '))

                res=','.join([str(i) for i in res_x1])
                f2.write(res+'\n')

if __name__ == '__main__':
    # obtain_data(0,10000000,"train","1000M")
    # obtain_data(0,10000,"train","10000")
    # obtain_valid_test_data(0,1000,"valid","1000")
    # obtain_valid_test_data(0,10000,"test","10000")

    # process_oeis_data(30000, "valid",25)
    # process_oeis_data(30000, "valid",35)
    # delete_same_data()
    src="+ 0 + 1 + 14 + 54 + 136 + 275 + 486 + 784 + 1184 + 1701 + 2350 + 3146 + 4104 + 5239 + 6566 + 8100 + 9856 + 1 1849 + 1 4094 + 1 6606 + 1 9400 + 2 2491 + 2 5894 + 2 9624 + 3 3696 + 3 8125 + 4 2926 + 4 8114 + 5 3704 + 5 9711"
    src=src.split(' ')


    print(decode(src))

    # trans_test()