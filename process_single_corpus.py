import pickle
from collections import Counter

'''
@method: load_pickle
    @description: 加载pickle文件并返回加载的对象。
    @package: process_single_corpus
    @param filename (str): pickle文件的路径。
    @return: 加载的对象。
    @analysis:
        @original: 代码逻辑
        @explanation:
            以二进制模式和编码'iso-8859-1'打开pickle文件。
            使用pickle.load()函数从文件中加载对象。
            返回加载的对象。

@method: single_list
    @description: 统计给定列表中目标值的出现次数。
    @package: process_single_corpus
    @param arr (list): 需要搜索的列表。
    @param target: 需要统计出现次数的目标值。
    @return: 目标值在列表中的出现次数。
    @analysis:
        @original: 代码逻辑
        @explanation:
            使用列表的count()方法统计目标值在列表中出现的次数。
            返回统计结果。
'''
def load_pickle(filename):
    return pickle.load(open(filename, 'rb'), encoding='iso-8859-1')
def single_list(arr, target):
    return arr.count(target)

'''
@method: data_staqc_prpcessing
    @description: 将语料分隔为单个和多个部分，并保存到指定文件路径
    @param filepath: 原始数据文件路径
    @param save_single_path: 保存单个部分的文件路径
    @param save_mutiple_path: 保存多个部分的文件路径
    @return: None (无返回值)
    @analysis:
        @original: 代码逻辑
        @explanation:
            从指定文件路径中读取数据，存储在total_data中
            提取所有qid，存储在qids列表中
            统计每个qid的出现次数，保存在result中
            创建空的total_data_single和total_data_multiple列表
            遍历total_data，根据每个qid的出现次数将数据拆分为单个答案和多个答案部分，并分别添加到对应列表中
            将单个部分保存到指定文件路径save_single_path
            将多个部分保存到指定文件路径save_mutiple_path
    @changes: 无
'''
#staqc：把语料中的单候选和多候选分隔开
def data_staqc_prpcessing(filepath,save_single_path,save_mutiple_path):
    # 从文件中读取数据
    with open(filepath,'r')as f:
        total_data= eval(f.read())
        f.close()
    # 提取所有qid
    qids = []
    for i in range(0, len(total_data)):
        qids.append(total_data[i][0][0])

    # 将数据拆分为单个和多个结果的部分
    result = Counter(qids)

    total_data_single = []
    total_data_multiple = []
    for i in range(0, len(total_data)):
        if(result[total_data[i][0][0]]==1):
            total_data_single.append(total_data[i])

        else:
            total_data_multiple.append(total_data[i])

    # 将单个结果部分写入到保存文件
    f = open(save_single_path, "w")
    f.write(str(total_data_single))
    f.close()

    # 将多个结果部分保存到文件
    f = open(save_mutiple_path, "w")
    f.write(str(total_data_multiple))
    f.close()


'''
@method: data_large_prpcessing
    @description: 将语料分隔为单个和多个部分，并保存到指定文件路径。
    @param filepath (str): 原始数据文件的路径。
    @param save_single_path (str): 保存单个部分的文件路径。
    @param save_mutiple_path (str): 保存多个部分的文件路径。
    @return: None (无返回值)。
    @analysis:
        @original: 代码逻辑。
        @explanation:
            加载pickle文件，将其存储在total_data中。
            创建一个空列表qids。
            打印total_data的长度。
            遍历total_data，将每个数据项的qid（即total_data[i][0][0]）添加到qids列表中。
            打印qids的长度。
            使用Counter函数统计qids列表中每个qid出现的次数，结果保存在result中。
            创建空列表total_data_single和total_data_multiple，分别用于存储单个部分和多个部分的数据。
            遍历total_data，根据每个数据项的qid在result中的出现次数，将数据分别添加到total_data_single和total_data_multiple中。
            打印total_data_single的长度。
            将total_data_single以二进制写入到save_single_path文件中。
            将total_data_multiple以二进制写入到save_mutiple_path文件中。
    @changes: 无。
'''
# 从文件中加载数据
#large:把语料中的但候选和多候选分隔开
def data_large_prpcessing(filepath,save_single_path,save_mutiple_path):
    total_data = load_pickle(filepath)
    qids = []
    print(len(total_data))
    for i in range(0, len(total_data)):
        qids.append(total_data[i][0][0])
    print(len(qids))
    result = Counter(qids)
    total_data_single = []
    total_data_multiple = []
    for i in range(0, len(total_data)):
        if (result[total_data[i][0][0]] == 1 ):
            total_data_single.append(total_data[i])
        else:
            total_data_multiple.append(total_data[i])
    print(len(total_data_single))


    with open(save_single_path, 'wb') as f:
        pickle.dump(total_data_single, f)
    with open(save_mutiple_path, 'wb') as f:
        pickle.dump(total_data_multiple, f)


"""
@method: single_unlable2lable
    @description: 将单个候选的数据转换为带有标签的格式，仅保留qid信息。
    @param path1 (str): 原始数据文件的路径。
    @param path2 (str): 转换后的数据文件的路径。
    @return: None 无返回值。
    @analysis:
        @original: 代码逻辑。
        @explanation:
            加载原始数据文件，存储为total_data。
            创建一个空的标签列表labels。
            遍历total_data，将每个数据的qid和标签1添加到labels中。
            根据qid对labels进行排序。
            将排序后的结果写入转换后的数据文件。
    @changes: 无。
"""
#把单候选只保留其qid
def single_unlable2lable(path1,path2):
    total_data = load_pickle(path1)
    labels=[]

    for i in range(0,len(total_data)):
        labels.append([total_data[i][0],1])

    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))
    f = open(path2, "w")
    f.write(str(total_data_sort))
    f.close()


if __name__ == "__main__":
    #将staqc_python中的单候选和多候选分开
    staqc_python_path = '../hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_sigle_save ='../hnn_process/ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = '../hnn_process/ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    #data_staqc_prpcessing(staqc_python_path,staqc_python_sigle_save,staqc_python_multiple_save)

    #将staqc_sql中的单候选和多候选分开
    staqc_sql_path = '../hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_sigle_save = '../hnn_process/ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = '../hnn_process/ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    #data_staqc_prpcessing(staqc_sql_path, staqc_sql_sigle_save, staqc_sql_multiple_save)

    #将large_python中的单候选和多候选分开
    large_python_path = '../hnn_process/ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'
    large_python_single_save = '../hnn_process/ulabel_data/large_corpus/single/python_large_single.pickle'
    large_python_multiple_save ='../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    data_large_prpcessing(large_python_path, large_python_single_save, large_python_multiple_save)

    # 将large_sql中的单候选和多候选分开
    large_sql_path = '../hnn_process/ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'
    large_sql_single_save = '../hnn_process/ulabel_data/large_corpus/single/sql_large_single.pickle'
    large_sql_multiple_save = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    #data_large_prpcessing(large_sql_path, large_sql_single_save, large_sql_multiple_save)

    large_sql_single_label_save = '../hnn_process/ulabel_data/large_corpus/single/sql_large_single_label.txt'
    large_python_single_label_save = '../hnn_process/ulabel_data/large_corpus/single/python_large_single_label.txt'
    #single_unlable2lable(large_sql_single_save,large_sql_single_label_save)
    #single_unlable2lable(large_python_single_save, large_python_single_label_save)
