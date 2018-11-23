import csv
import json
import random
random.seed(1)

def parse_csv():
    class_to_ids = {}
    for i in range(28):
        class_to_ids[i] = []

    with open('all.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            else:
                labels = [int(n) for n in row[1].split(' ')]
                for l in labels:
                    class_to_ids[l].append(row[0])
                line_count += 1
        print(f'Processed {line_count} lines.')

    # 将属于每个类的图片id存入txt文件，每行一个id
    for i in range(28):
        with open('preprocess/{}.txt'.format(i), 'w+') as f:
            f.write("\n".join(class_to_ids[i]))
            print("There are {0} numbers pictures with label {1}".format(len(class_to_ids[i]), i))

    # 计算每个类出现的概率，存入prob.json
    dict = {}
    for i in range(28):
        dict[i] = len(class_to_ids[i]) * 1.0 / line_count
    with open('prob.json', 'w+') as f:
        f.write(json.dumps(dict))


def dataset_split(train_percent=0.33, validation_percent=0.33, test_percent=0.33):
    content = []
    with open('all.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] == 'Id':
                continue
            content.append(",".join(row))
    random.shuffle(content)
    total_num = len(content)
    with open('train.csv', 'w+') as f:
        train_num = int(total_num * train_percent)
        f.write("\n".join(content[0:train_num]))
    with open('validation.csv', 'w+') as f:
        valid_num = int(total_num * validation_percent)
        f.write("\n".join(content[train_num: train_num + valid_num]))
    with open('test.csv', 'w+') as f:
        f.write("\n".join(content[train_num + valid_num: -1]))


if __name__ == '__main__':
    # parse_csv()
    dataset_split()
