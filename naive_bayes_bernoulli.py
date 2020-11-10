import numpy as np
import text_preprocessing as tp

classes_num = tp.classes_num


def train_nb(train_matrix, train_category):
    p_class = [0] * classes_num
    for i in range(len(p_class)):
        # laplace smoothing
        p_class[i] = np.log((train_category.count(i) + 1.0) / (len(train_category) + classes_num))

    num_words = len(train_matrix[0])
    p_words_num = []
    p_words_denom = []
    p_words = []
    p_words_not = []

    for i in range(classes_num):
        # laplace smoothing
        p_words_num.append(np.ones(num_words))
        p_words_denom.append(train_category.count(i) + 2)

    for i in range(len(train_matrix)):
        p_words_num[train_category[i]] += train_matrix[i]

    for i in range(classes_num):
        p_words.append(np.log(p_words_num[i] / p_words_denom[i]))
        p_words_not.append(np.log(1.0 - (p_words_num[i] / p_words_denom[i])))

    return p_words, p_words_not, p_class


def get_max_probability_category(p_words, p_words_not, p_class, bag_of_words_vec):
    probability = []

    for class_index in range(classes_num):
        log_sum = p_class[class_index]
        for i in range(len(bag_of_words_vec)):
            if bag_of_words_vec[i] > 0:
                log_sum += p_words[class_index][i]
            else:
                log_sum += p_words_not[class_index][i]
        probability.append(log_sum)

    return np.argmax(probability)


def nb_classify(p_words, p_words_not, p_class, test_matrix, test_true_category, vocab_list, docs_list):
    acc = 0
    for i in range(len(test_matrix)):
        # print(get_words_by_bow(vocab_list, test_matrix[i]))
        predict_class = get_max_probability_category(p_words, p_words_not, p_class, test_matrix[i])
        if predict_class == test_true_category[i]:
            acc += 1
            #print('正确,类别:' + tp.classes_name[predict_class])
        else:
            print(get_words_by_bow(vocab_list, test_matrix[i]))
            print(docs_list[i])
            print('错误,预测类别:' + tp.classes_name[predict_class] + ',实际类别:' + tp.classes_name[test_true_category[i]])
            print()
    print('准确率:%.2f%%' % (100.0 * acc / len(test_matrix)))


def get_words_by_bow(vocab_list, bow_vec):
    words_list = []
    for i in range(len(vocab_list)):
        for j in range(bow_vec[i]):
            words_list.append(vocab_list[i])
    return words_list


def main():
    vocab_list = tp.load_vocab_list(tp.vocab_list_dir)
    print('词库数:' + str(len(vocab_list)))
    train_matrix = tp.load_train_matrix(tp.train_set_of_words_dir)

    train_category = tp.load_train_category(tp.train_class_dir)

    p_words, p_words_not, p_class = train_nb(train_matrix, train_category)

    test_matrix = tp.load_test_matrix(tp.test_bag_of_words_dir)
    test_true_category = tp.load_test_category(tp.test_class_dir)
    docs_list = tp.load_docs_list(tp.test_docs_list_dir)

    print('测试集数:' + str(len(test_true_category)))

    nb_classify(p_words, p_words_not, p_class, test_matrix, test_true_category, vocab_list, docs_list)


if __name__ == '__main__':
    main()
