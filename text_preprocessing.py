import os
import re

classes_index = {'政治': 0, '经济': 1, '教育': 2, '法律': 3, '电脑': 4, '体育': 5}
classes_name = ['政治', '经济', '教育', '法律', '电脑', '体育']
classes_num = 6

vocab_list_dir = 'preprocessing_data/vocab_list.txt'

train_set_of_words_dir = 'preprocessing_data/train_set_of_words.txt'
train_bag_of_words_dir = 'preprocessing_data/train_bag_of_words.txt'
train_class_dir = 'preprocessing_data/train_class.txt'
train_docs_list_dir = 'preprocessing_data/train_docs_list.txt'

test_set_of_words_dir = 'preprocessing_data/test_set_of_words.txt'
test_bag_of_words_dir = 'preprocessing_data/test_bag_of_words.txt'
test_class_dir = 'preprocessing_data/test_class.txt'
test_docs_list_dir = 'preprocessing_data/test_docs_list.txt'


def load_data_set(is_train_set):
    if is_train_set:
        base_set_dir = 'data/train'
    else:
        base_set_dir = 'data/test'

    all_path = []
    for dirname in os.listdir(base_set_dir):
        filename = os.path.join(base_set_dir, dirname)
        all_path.append(filename)

    data_list, docs_list = get_data_list(all_path)
    vocab_list = create_vocab_list(data_list)
    return vocab_list, data_list, docs_list


def get_words_class_vec(vocab_list, data_list, is_set_of_words):
    classes_list = []
    words_list = []

    count = 0
    for i in range(len(data_list)):
        classes_list.extend([i] * len(data_list[i]))
        for j in range(len(data_list[i])):
            print(data_list[i][j])
            count += 1
            print('当前进度:%.2f%%' % (100.0 * (count / (len(data_list[0]) + len(data_list[1]) +
                                len(data_list[2]) + len(data_list[3]) + len(data_list[4]) + len(data_list[5])))))

            if is_set_of_words:
                words_list.append(set_of_words2vec(vocab_list, data_list[i][j]))
            else:
                words_list.append(bag_of_words2vec(vocab_list, data_list[i][j]))

    return words_list, classes_list


def get_data_list(paths):
    docs_list = [list()] * classes_num
    data_list = [list()] * classes_num
    for path in paths:
        class_name = path[-6:-4]
        class_index = classes_index[class_name]
        # print(class_name, class_index)
        with open(path, encoding='UTF-8') as f:
            class_list = []
            doc_list = []
            pattern = re.compile(r'<text>([\w\W]*?)</text>')
            all_docs = pattern.findall(f.read())

            for doc in all_docs:
                class_list.append(list(filter(lambda x: len(x) != 0, re.split(r'[\s]+', doc))))
                # print(doc)
                doc_list.append(re.sub(r'[\s]+', '', doc))

            data_list[class_index] = class_list
            docs_list[class_index] = doc_list

    return data_list, docs_list


def create_vocab_list(data_list):
    vocab_set = set()
    for class_list in data_list:
        for doc_list in class_list:
            vocab_set = vocab_set | set(doc_list)

    stop_words_list = load_stop_words_zh()
    vocab_set = vocab_set - set(stop_words_list)
    return sorted(list(vocab_set))


def load_stop_words_zh():
    filename = 'data/stop_words_zh.txt'
    stop_words_list = []

    with open(filename, encoding='UTF-8') as f:
        for line in f.readlines():
            stop_words_list.append(line.strip())
    return stop_words_list


def set_of_words2vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
    return return_vec


def bag_of_words2vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def write_words_file(words_list, filename):
    with open(filename, 'w') as f:
        for word in words_list:
            for i in range(len(word)):
                if i != 0:
                    f.write(' ' + str(word[i]))
                else:
                    f.write(str(word[i]))
            f.write('\n')


def write_class_file(class_list, filename):
    with open(filename, 'w') as f:
        for i in range(len(class_list)):
            if i != 0:
                f.write(' ' + str(class_list[i]))
            else:
                f.write(str(class_list[i]))


def write_vocab_file(vocab_list, filename):
    with open(filename, encoding='UTF-8', mode='w') as f:
        for vocab in vocab_list:
            f.write(str(vocab) + '\n')


def write_docs_file(docs_list, filename):
    with open(filename, encoding='UTF-8', mode='w') as f:
        for class_docs in docs_list:
            for doc in class_docs:
                f.write(str(doc) + '\n')


# 从文件获取预处理的数据结构
def load_vocab_list(filename):
    vocab_list = []
    with open(filename, encoding='UTF-8') as f:
        for line in f.readlines():
            vocab_list.append(line.strip())
    return vocab_list


def load_train_matrix(filename):
    train_matrix = []
    with open(filename, encoding='UTF-8') as f:
        for line in f.readlines():
            set_of_words = list(map(lambda x: int(x), line.strip().split(' ')))
            train_matrix.append(set_of_words)
    return train_matrix


def load_train_category(filename):
    with open(filename, encoding='UTF-8') as f:
        train_category = list(map(lambda x: int(x), f.read().strip().split(' ')))
    return train_category


def load_test_matrix(filename):
    test_matrix = []
    with open(filename, encoding='UTF-8') as f:
        for line in f.readlines():
            bag_of_words = list(map(lambda x: int(x), line.strip().split(' ')))
            test_matrix.append(bag_of_words)
    return test_matrix


def load_docs_list(filename):
    docs_list = []
    with open(filename, encoding='UTF-8') as f:
        for line in f.readlines():
            docs_list.append(line.strip())
    return docs_list


def load_test_category(filename):
    with open(filename, encoding='UTF-8') as f:
        test_category = list(map(lambda x: int(x), f.read().strip().split(' ')))
    return test_category


def main():
    # 训练集预处理
    train_vocab_list, train_data_list, train_docs_list = load_data_set(True)
    # write_vocab_file(train_vocab_list, vocab_list_dir)
    # write_docs_file(train_docs_list, train_docs_list_dir)
    # train_set_of_words_list, train_class_list = get_words_class_vec(train_vocab_list, train_data_list, True)
    # write_words_file(train_set_of_words_list, train_set_of_words_dir)
    # write_class_file(train_class_list, train_class_dir)
    # train_bag_of_words_list, train_class_list = get_words_class_vec(train_vocab_list, train_data_list, False)
    # write_words_file(train_bag_of_words_list, train_bag_of_words_dir)
    #
    # # 测试集预处理
    test_vocab_list, test_data_list, test_docs_list = load_data_set(False)
    # write_docs_file(test_docs_list, test_docs_list_dir)
    # test_bag_of_words_list, test_class_list = get_words_class_vec(train_vocab_list, test_data_list, False)
    # write_words_file(test_bag_of_words_list, test_bag_of_words_dir)
    # write_class_file(test_class_list, test_class_dir)
    test_set_of_words_list, test_class_list = get_words_class_vec(train_vocab_list, test_data_list, True)
    write_words_file(test_set_of_words_list, test_set_of_words_dir)


if __name__ == '__main__':
    main()


