import numpy as np


class TextConverter(object):
    def __init__(self, corpus, max_vocab=5000):
        """
        建立一个字符索引转换器
            Args:
                corpus: 语料库
                max_vocab: 最大的单词数量
        """
        corpus = corpus.replace('\n', ' ').replace('\r', ' ').replace('，', ' ').replace('。', ' ').replace('《', ' ').replace('》', ' ')
        # 去掉重复字符
        vocab = set(corpus)
        # 如果单词总数超过最大数值，去掉频率最低的
        vocab_count = {}

        # 计算单词出现频率并排序
        for word in vocab:
            vocab_count[word] = 0
        for word in corpus:
            vocab_count[word] += 1
        vocab_count_list = []
        for word in vocab_count:
            vocab_count_list.append((word, vocab_count[word]))
        vocab_count_list.sort(key=lambda x: x[1], reverse=True)
        # 如果超过最大值，截取频率最低的字符
        if len(vocab_count_list) > max_vocab:
            vocab_count_list = vocab_count_list[:max_vocab]
            
        vocab = [x[0] for x in vocab_count_list]
        #上边已经将vocab排好序
        self.vocab = vocab
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))
        # vocab_size = len(self.vocab) + 1

    @ property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<UNK>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            return Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

