import csv


def parse_csv():
    class_to_ids = {}
    for i in range(28):
        class_to_ids[i] = []

    with open('train.csv') as csv_file:
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

    for i in range(28):
        with open('preprocess/{}.txt'.format(i), 'w+') as f:
            f.write("\n".join(class_to_ids[i]))
            print("There are {0} numbers pictures with label {1}".format(len(class_to_ids[i]), i))
