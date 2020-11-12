import text_preprocessing as tp
import numpy as np
# from scipy.special import softmax as sf


class Softmax:
    # features = [[X1], [X2], ..., [Xn]], n is num of samples
    # x1([x1, x2, ..., xm]) is bow vec of sample #1, m is num of features
    # labels = [0, 0, ..., 1, 1, ..., 5, 5, 5], size(labels) = n, c is num of classes
    # weights = [[w1], [w2], ..., [wm]]
    def __init__(self, features, labels, num_of_classes, alpha=0.01, iterations=50):
        self.alpha = alpha
        self.iterations = iterations
        self.features = np.array(features)
        self.labels = np.array(labels)

        self.n, self.m = np.shape(self.features)
        self.c = num_of_classes

        self.weights = np.zeros((self.c, self.m), dtype=np.float)

    @staticmethod
    def soft(X):
        # X = features * weights.T
        # print(X)
        x_exp = np.exp(X)
        x_sum = np.sum(x_exp, axis=1, keepdims=True)  # 保持二维特性
        return x_exp / x_sum
        # return sf(X)

    def gradient(self, k, s):
        sum_grad = np.zeros(self.m)
        for i in range(self.n):
            if self.labels[i] == k:
                sum_grad += ((1 - s[i][k]) * self.features[i])
            else:
                sum_grad += ((0 - s[i][k]) * self.features[i])
        return sum_grad

    def train(self):
        for i in range(self.iterations):
            print('training, ' + str(i) + 'th')
            tmp_weights = np.zeros((self.c, self.m), dtype=np.float)
            for k in range(self.c):
                X = np.dot(self.features, self.weights.T)
                Max = np.max(X, axis=1, keepdims=True)
                X = X - Max
                tmp_weights[k] = self.alpha * self.gradient(k, self.soft(X))
            self.weights += tmp_weights

    def test(self, test_features):
        # predict_soft = self.soft(test_features @ self.weights.T)
        predict_soft = test_features @ self.weights.T
        # print(predict_soft)
        return predict_soft.argmax(axis=1)


def get_words_by_bow(vocab_list, bow_vec):
    words_list = []
    for i in range(len(vocab_list)):
        for j in range(bow_vec[i]):
            words_list.append(vocab_list[i])
    return words_list


def acc_count(test_matrix, test_predict_category, test_true_category, vocab_list, docs_list):
    acc = 0
    for i in range(len(test_predict_category)):
        if test_predict_category[i] == test_true_category[i]:
            acc += 1
            #print('正确,类别:' + tp.classes_name[predict_class])
        else:
            print(get_words_by_bow(vocab_list, test_matrix[i]))
            print(docs_list[i])
            print('错误,预测类别:' + tp.classes_name[test_predict_category[i]] + ',实际类别:' + tp.classes_name[test_true_category[i]])
            print()
    print('准确率:%.2f%%' % (100.0 * acc / len(test_matrix)))


def presence_run():
    vocab_list = tp.load_vocab_list(tp.vocab_list_dir)
    print('词库数:' + str(len(vocab_list)))
    # train_matrix = tp.load_train_matrix(tp.train_bag_of_words_dir)
    train_matrix = tp.load_train_matrix(tp.train_set_of_words_dir)
    train_matrix = np.insert(train_matrix, 0, 1, axis=1)

    train_category = tp.load_train_category(tp.train_class_dir)

    softmax = Softmax(train_matrix, train_category, tp.classes_num, 0.01, 50)
    softmax.train()

    # test_matrix = tp.load_test_matrix(tp.test_bag_of_words_dir)
    test_matrix = tp.load_test_matrix(tp.test_set_of_words_dir)
    test_matrix = np.insert(test_matrix, 0, 1, axis=1)

    test_true_category = tp.load_test_category(tp.test_class_dir)
    docs_list = tp.load_docs_list(tp.test_docs_list_dir)
    print('测试集数:' + str(len(test_true_category)))

    test_predict_category = softmax.test(test_matrix)

    acc_count(test_matrix, test_predict_category, test_true_category, vocab_list, docs_list)


def frequency_run():
    vocab_list = tp.load_vocab_list(tp.vocab_list_dir)
    print('词库数:' + str(len(vocab_list)))
    train_matrix = tp.load_train_matrix(tp.train_bag_of_words_dir)
    train_matrix = np.insert(train_matrix, 0, 1, axis=1)
    # train_matrix = tp.load_train_matrix(tp.train_set_of_words_dir)
    train_category = tp.load_train_category(tp.train_class_dir)

    softmax = Softmax(train_matrix, train_category, tp.classes_num, 0.01, 50)
    softmax.train()

    test_matrix = tp.load_test_matrix(tp.test_bag_of_words_dir)
    test_matrix = np.insert(test_matrix, 0, 1, axis=1)

    # test_matrix = tp.load_test_matrix(tp.test_set_of_words_dir)
    test_true_category = tp.load_test_category(tp.test_class_dir)
    docs_list = tp.load_docs_list(tp.test_docs_list_dir)
    print('测试集数:' + str(len(test_true_category)))

    test_predict_category = softmax.test(test_matrix)

    acc_count(test_matrix, test_predict_category, test_true_category, vocab_list, docs_list)


def main():
    print('Choose classification based on presence(0) or frequency(1): ', end='')
    choose = int(input())
    if choose == 0:
        presence_run()
    elif choose == 1:
        frequency_run()
    else:
        print('wrong input!')


if __name__ == '__main__':
    main()
