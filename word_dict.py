"""
@method: get_vocab
    @package: word_dict.py
    @description: 获取语料库中的词汇表。
    @param corpus1 (list): 语料库1。
    @param corpus2 (list): 语料库2。
    @return: word_vacab (set): 词汇表集合。
    @analysis:
        @original: 代码逻辑
        @explanation:
            1. 初始化一个空的词汇表集合 word_vacab。
            2. 遍历 corpus1 中的每个元素：
               - 对于 corpus1[i][1][0] 中的每个词，将其添加到 word_vacab 中。
               - 对于 corpus1[i][1][1] 中的每个词，将其添加到 word_vacab 中。
               - 对于 corpus1[i][2][0] 中的每个词，将其添加到 word_vacab 中。
               - 对于 corpus1[i][3] 中的每个词，将其添加到 word_vacab 中。
            3. 遍历 corpus2 中的每个元素：
               - 对于 corpus2[i][1][0] 中的每个词，将其添加到 word_vacab 中。
               - 对于 corpus2[i][1][1] 中的每个词，将其添加到 word_vacab 中。
               - 对于 corpus2[i][2][0] 中的每个词，将其添加到 word_vacab 中。
               - 对于 corpus2[i][3] 中的每个词，将其添加到 word_vacab 中。
            4. 打印词汇表的长度。
            5. 返回词汇表集合 word_vacab。
    @changes: 无
"""
#构建初步词典的具体步骤1
def get_vocab(corpus1,corpus2):
    word_vacab = set()
    for i in range(0,len(corpus1)):
        for j in range(0,len(corpus1[i][1][0])):
            word_vacab.add(corpus1[i][1][0][j])
        for j in range(0,len(corpus1[i][1][1])):
            word_vacab.add(corpus1[i][1][1][j])
        for j in range(0,len(corpus1[i][2][0])):#len(corpus2[i][2])
            word_vacab.add(corpus1[i][2][0][j])#注意之前是 word_vacab.add(corpus2[i][2][j])
        for j in range(0,len(corpus1[i][3])):
            word_vacab.add(corpus1[i][3][j])

    for i in range(0,len(corpus2)):
        for j in range(0,len(corpus2[i][1][0])):
            word_vacab.add(corpus2[i][1][0][j])
        for j in range(0,len(corpus2[i][1][1])):
            word_vacab.add(corpus2[i][1][1][j])
        for j in range(0,len(corpus2[i][2][0])):#len(corpus2[i][2])
            word_vacab.add(corpus2[i][2][0][j])#注意之前是 word_vacab.add(corpus2[i][2][j])
        for j in range(0,len(corpus2[i][3])):
            word_vacab.add(corpus2[i][3][j])
    print(len(word_vacab))
    return word_vacab

import pickle

#读取picle
def load_pickle(filename):
    return pickle.load(open(filename, 'rb'), encoding='iso-8859-1')

'''
@method: vocab_prpcessing
    @package: word_dict.py
    @description: 处理构建初步词典的过程。
    @param filepath1 (str): 文件路径1，包含语料库1的数据。
    @param filepath2 (str): 文件路径2，包含语料库2的数据。
    @param save_path (str): 保存路径，用于保存构建的词典。
    @analysis:
        @original: 代码逻辑
        @explanation:
            1. 使用 "with open" 语句打开 filepath1 文件，并读取其中的数据并使用 eval 函数进行转换。
               将转换后的数据存储在 total_data1 变量中。
            2. 使用 "with open" 语句打开 filepath2 文件，并读取其中的数据并使用 eval 函数进行转换。
               将转换后的数据存储在 total_data2 变量中。
            3. 调用 get_vocab 方法，传入 total_data2 和 total_data2 作为参数，获取词汇表 x1。
            4. 以写入模式打开 save_path 文件，并将词汇表 x1 转换为字符串写入文件中。
            5. 关闭文件。
    @changes: 无
'''
#构建初步词典
def vocab_prpcessing(filepath1,filepath2,save_path):
    with open(filepath1, 'r')as f:
        total_data1 = eval(f.read())
        f.close()

    with open(filepath2, 'r')as f:
        total_data2 = eval(f.read())
        f.close()

    x1= get_vocab(total_data2,total_data2)
    #total_data_sort = sorted(x1, key=lambda x: (x[0], x[1]))
    f = open(save_path, "w")
    f.write(str(x1))
    f.close()

'''
@method: final_vocab_prpcessing
    @package: <unknown>
    @description: 处理最终构建词典的过程。
    @param filepath1 (str): 文件路径1，包含语料库1的数据。
    @param filepath2 (str): 文件路径2，包含语料库2的数据。
    @param save_path (str): 保存路径，用于保存构建的词典。
    @analysis:
        @original: 代码逻辑
        @explanation:
            1. 创建一个空集合 word_set，用于存储最终的词汇表。
            2. 使用 "with open" 语句打开 filepath1 文件，并读取其中的数据并使用 eval 函数进行转换。
               将转换后的数据存储在 total_data1 变量中，同时转换为集合类型。
            3. 使用 "with open" 语句打开 filepath2 文件，并读取其中的数据并使用 eval 函数进行转换。
               将转换后的数据存储在 total_data2 变量中。
            4. 将 total_data1 转换为列表类型。
            5. 调用 get_vocab 方法，传入 total_data2 和 total_data2 作为参数，获取词汇表 x1。
            6. 遍历词汇表 x1 中的每个词汇，判断其是否存在于 total_data1 中。
               如果存在，则跳过继续下一个词汇；如果不存在，则将其添加到 word_set 集合中。
            7. 打印 total_data1 的长度和 word_set 的长度。
            8. 打开 save_path 文件以写入模式，并将词汇表 word_set 转换为字符串写入文件中。
            9. 关闭文件。
    @changes: 无
'''
def final_vocab_prpcessing(filepath1,filepath2,save_path):
    word_set = set()
    with open(filepath1, 'r')as f:
        total_data1 = set(eval(f.read()))
        f.close()
    with open(filepath2, 'r')as f:
        total_data2 = eval(f.read())
        f.close()
    total_data1 = list(total_data1)
    x1= get_vocab(total_data2,total_data2)
    #total_data_sort = sorted(x1, key=lambda x: (x[0], x[1]))
    for i in x1:
        if i in total_data1:
            continue
        else:
            word_set.add(i)
    print(len(total_data1))
    print(len(word_set))
    f = open(save_path, "w")
    f.write(str(word_set))
    f.close()




if __name__ == "__main__":
    #====================获取staqc的词语集合===============
    python_hnn = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/python_hnn_data_teacher.txt'
    python_staqc = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/python_staqc_data.txt'
    python_word_dict = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/python_word_vocab_dict.txt'

    sql_hnn = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/sql_hnn_data_teacher.txt'
    sql_staqc = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/sql_staqc_data.txt'
    sql_word_dict = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/sql_word_vocab_dict.txt'

    # vocab_prpcessing(python_hnn,python_staqc,python_word_dict)
    # vocab_prpcessing(sql_hnn,sql_staqc,sql_word_dict)
    #====================获取最后大语料的词语集合的词语集合===============
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'
    final_vocab_prpcessing(sql_word_dict, new_sql_large, large_word_dict_sql)
    #vocab_prpcessing(new_sql_staqc,new_sql_large,final_word_dict_sql)

    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large ='../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    #final_vocab_prpcessing(python_word_dict, new_python_large, large_word_dict_python)
    #vocab_prpcessing(new_python_staqc,new_python_large,final_word_dict_python)





