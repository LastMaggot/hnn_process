
'''

并行分词
'''
import os
import pickle
import logging

import sys
sys.path.append("..")

# 解析结构
from python_structured import *
from sqlang_structured import *

#FastText库  gensim 3.4.0
from gensim.models import FastText

import numpy as np

#词频统计库
import collections
#词云展示库
import wordcloud
#图像处理库 Pillow 5.1.0
from PIL import Image

# 多进程
from multiprocessing import Pool as ThreadPool


'''
本文件函数名关键字说明：
@keywords:
    @prefix:multipro
        @explanation: 多线程执行中调用的功能函数
'''

"""
@method: multipro_python_query
    @description:  对数据列表中的每一行，进行 Python 查询解析
    @param: data_list: 包含待解析数据的列表
    @return: 包含解析结果的列表
    """
#python解析
def multipro_python_query(data_list):
    result=[python_query_parse(line) for line in data_list]
    return result

"""
@method: multipro_python_code
    @description: 对数据列表中的每个元素进行 Python 代码解析
    @param: data_list: 包含待解析数据的列表
    @return: 包含解析结果的列表
    """

'''
@method: multipro_python_code
    @description: 对给定的数据列表中的每个元素进行 Python 代码解析
    @param: data_list: 数据列表，包含多个元素
    @return: result: 解析结果列表
    @analysis:
        @original:函数逻辑
        @explanation:
        创建一个空列表result用于存储解析结果
        遍历数据列表data_list中的每个元素
        调用python_code_parse函数对当前元素进行解析，并将解析结果添加到result列表中
        返回result列表作为解析结果
'''
def multipro_python_code(data_list):
    result = [python_code_parse(line) for line in data_list]
    return result

'''
@method: multipro_python_context
    @description: 对给定的数据列表中的每个元素进行 Python 上下文解析
    @param: data_list: 数据列表，包含多个元素
    @return: result: 解析结果列表
    @analysis:
        @original:函数逻辑
        @explanation:
        创建一个空列表result用于存储解析结果
        遍历数据列表data_list中的每个元素
        判断当前元素是否为'-10000'
        如果是，则将字符串'-10000'作为解析结果添加到result列表中
        否则，调用python_context_parse函数对当前元素进行解析，并将解析结果添加到result列表中
        返回result列表作为解析结果
'''
def multipro_python_context(data_list):
    result = []
    for line in data_list:
        if (line == '-10000'):
            result.append(['-10000'])
        else:
            result.append(python_context_parse(line))
    return result


#下面是SQL解析部分：
'''
multipro_sqlang_query(data_list)：
    该函数用于并行处理给定的SQL语言查询列表，并返回处理结果。
    它将每个查询传递给sqlang_query_parse函数进行解析，并将解析结果添加到结果列表中。
multipro_sqlang_code(data_list)：
    该函数用于并行处理给定的SQL语言代码列表，并返回处理结果。
    它将每个代码传递给sqlang_code_parse函数进行解析，并将解析结果添加到结果列表中。
multipro_sqlang_context(data_list)：
    该函数用于并行处理给定的SQL语言上下文列表，并返回处理结果。
    它根据每个上下文的内容调用sqlang_context_parse函数进行解析，并将解析结果添加到结果列表中。
'''


'''
@method: multipro_sqlang_query
    @description: 该函数用于并行处理给定的SQL语言查询列表，并返回处理结果。
    @param: data_list: 数据列表，包含多个元素
    @return: result: 解析结果列表
    @analysis:
        @original:函数逻辑
        @explanation:
        创建一个空列表result用于存储解析结果
        遍历数据列表data_list中的每个元素
        调用sqlang_query_parse函数对当前元素进行解析，并将解析结果添加到result列表中
        返回result列表作为解析结果
'''


#sql解析
def multipro_sqlang_query(data_list):
    result=[sqlang_query_parse(line) for line in data_list]
    return result


'''
@method: multipro_sqlang_code
    @description: 并行处理给定的SQL语言代码列表，并返回处理结果
    @param data_list: SQL语言代码列表
    @return: 处理结果列表
    @analysis:
        @original: 代码逻辑
        @explanation:
            遍历给定的SQL语言代码列表data_list
            若代码行为'-10000'，将['-10000']添加到结果列表result中
            否则，调用sqlang_code_parse函数对代码行进行处理，并将处理结果添加到结果列表result中
        @changes: 无

'''
def multipro_sqlang_code(data_list):
    result = [sqlang_code_parse(line) for line in data_list]
    return result

'''
@method: multipro_sqlang_context
    @description: 并行处理给定的SQL语言上下文列表，并返回处理结果
    @param data_list: SQL语言上下文列表
    @return: 处理结果列表
    @analysis:
        @original: 代码逻辑
        @explanation:
            遍历给定的SQL语言上下文列表data_list
            若上下文行为'-10000'，将['-10000']添加到结果列表result中
            否则，调用sqlang_context_parse函数对上下文行进行处理，并将处理结果添加到结果列表result中
        @changes: 无
'''
def multipro_sqlang_context(data_list):
    result = []
    for line in data_list:
        if (line == '-10000'):
            result.append(['-10000'])
        else:
            result.append(sqlang_context_parse(line))
    return result


"""
@method：parse_python
    @description: 解析Python列表数据，并且返回解析结果
    @package: getStructureToVector.py
    @param: python_list (list): 包含Python列表的数据列表。
    @param: split_num (int): 分割数，用于将数据列表分割成较小的子列表。
    Returns:
        tuple: 包含acont1_cut, acont2_cut, query_cut, code_cut和qids的元组。

    Analysis:
        @original: 代码逻辑
        @explanation:
            将python_list中的数据按照指定的split_num进行分割，并使用多线程池进行并行处理。
            对acont1数据进行处理并拼接结果到acont1_cut列表。
            对acont2数据进行处理并拼接结果到acont2_cut列表。
            对query数据进行处理并拼接结果到query_cut列表。
            对code数据进行处理并拼接结果到code_cut列表。
            获取qids列表，包含python_list中的所有qid值。
            打印第一个qid和qids的长度。
            返回acont1_cut, acont2_cut, query_cut, code_cut和qids的元组。
    """
def parse_python(python_list,split_num):

    acont1_data =  [i[1][0][0] for i in python_list]

    acont1_split_list = [acont1_data[i:i + split_num] for i in range(0, len(acont1_data), split_num)]
    pool = ThreadPool(10)
    acont1_list = pool.map(multipro_python_context, acont1_split_list)
    pool.close()
    pool.join()
    acont1_cut = []
    for p in acont1_list:
        acont1_cut += p
    print('acont1条数：%d' % len(acont1_cut))

    acont2_data = [i[1][1][0] for i in python_list]

    acont2_split_list = [acont2_data[i:i + split_num] for i in range(0, len(acont2_data), split_num)]
    pool = ThreadPool(10)
    acont2_list = pool.map(multipro_python_context, acont2_split_list)
    pool.close()
    pool.join()
    acont2_cut = []
    for p in acont2_list:
        acont2_cut += p
    print('acont2条数：%d' % len(acont2_cut))



    query_data = [i[3][0] for i in python_list]

    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multipro_python_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    code_data = [i[2][0][0] for i in python_list]

    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multipro_python_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))

    qids = [i[0] for i in python_list]
    print(qids[0])
    print(len(qids))

    return acont1_cut,acont2_cut,query_cut,code_cut,qids


"""
@method: parse_sqlang
    @description: 解析SQL语言列表数据，并返回解析结果
    @package:getStructureToVector
    @param: sqlang_list (list): 包含SQL语言列表的数据列表
    @param: split_num (int): 分割数，用于将数据列表分割成较小的子列表
    @return: tuple: 包含acont1_cut, acont2_cut, query_cut, code_cut和qids的元组

    @analysis:
        @original: 代码逻辑
        @explanation:
            从sqlang_list中提取acont1的数据，并存储在acont1_data列表中
            将acont1_data按照split_num进行分割，生成acont1_split_list列表
            使用多线程池ThreadPool创建线程池，并将acont1_split_list中的数据并行处理，结果存储在acont1_list中
            关闭并等待线程池中的线程执行完毕
            将acont1_list中的结果进行合并，存储在acont1_cut列表中
            打印合并后的acont1_cut列表的长度
            从sqlang_list中提取acont2的数据，并存储在acont2_data列表中
            将acont2_data按照split_num进行分割，生成acont2_split_list列表
            使用多线程池ThreadPool创建线程池，并将acont2_split_list中的数据并行处理，结果存储在acont2_list中
            关闭并等待线程池中的线程执行完毕
            将acont2_list中的结果进行合并，存储在acont2_cut列表中
            打印合并后的acont2_cut列表的长度
            从sqlang_list中提取query的数据，并存储在query_data列表中
            将query_data按照split_num进行分割，生成query_split_list列表
            使用多线程池ThreadPool创建线程池，并将query_split_list中的数据并行处理，结果存储在query_list中
            关闭并等待线程池中的线程执行完毕
            将query_list中的结果进行合并，存储在query_cut列表中
            打印合并后的query_cut列表的长度
            从sqlang_list中提取code的数据，并存储在code_data列表中
            将code_data按照split_num进行分割，生成code_split_list列表
            使用多线程池ThreadPool创建线程池，并将code_split_list中的数据并行处理，结果存储在code_list中
            关闭并等待线程池中的线程执行完毕
            将code_list中的结果进行合并，存储在code_cut列表中
            打印合并后的code_cut列表的长度
            从sqlang_list中提取qid的数据，存储在qids列表中
            打印qids列表的第一个元素和长度
            返回包含acont1_cut, acont2_cut, query_cut, code_cut和qids的元组
    """
def parse_sqlang(sqlang_list,split_num):

    acont1_data =  [i[1][0][0] for i in sqlang_list]

    acont1_split_list = [acont1_data[i:i + split_num] for i in range(0, len(acont1_data), split_num)]
    pool = ThreadPool(10)
    acont1_list = pool.map(multipro_sqlang_context, acont1_split_list)
    pool.close()
    pool.join()
    acont1_cut = []
    for p in acont1_list:
        acont1_cut += p
    print('acont1条数：%d' % len(acont1_cut))

    acont2_data = [i[1][1][0] for i in sqlang_list]

    acont2_split_list = [acont2_data[i:i + split_num] for i in range(0, len(acont2_data), split_num)]
    pool = ThreadPool(10)
    acont2_list = pool.map(multipro_sqlang_context, acont2_split_list)
    pool.close()
    pool.join()
    acont2_cut = []
    for p in acont2_list:
        acont2_cut += p
    print('acont2条数：%d' % len(acont2_cut))

    query_data = [i[3][0] for i in sqlang_list]

    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multipro_sqlang_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    code_data = [i[2][0][0] for i in sqlang_list]

    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multipro_sqlang_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))
    qids = [i[0] for i in sqlang_list]

    return acont1_cut ,acont2_cut,query_cut,code_cut,qids


"""
@method: main
    @description: 主函数，用于解析给定的语言类型的数据文件，并将解析结果存储到指定的文件中。
    @package:getStructureToVector
    @param lang_type (str): 语言类型，可选值为'python'和'sql'。
    @param split_num (int): 分割数，用于将数据列表分割成较小的子列表。
    @param source_path (str): 源数据文件的路径。
    @param save_path (str): 存储结果的文件路径。
    @return: None 无返回值
    @analysis:
        @original: 代码逻辑
        @explanation:
        创建一个空的总数据列表total_data
        打开源数据文件，加载数据到corpus_lis
        如果语言类型为'python'，则调用parse_python函数解析corpus_lis数据
            将解析结果逐个添加到total_data中
        如果语言类型为'sql'，则调用parse_sqlang函数解析corpus_lis数据
            将解析结果逐个添加到total_data中
        将total_data写入到存储文件中
        @changes: 无
"""
def main(lang_type,split_num,source_path,save_path):
    total_data = []
    with open(source_path, "rb") as f:
        #  存储为字典 有序
        # Ready
        corpus_lis  = pickle.load(f) #pickle

        #corpus_lis = eval(f.read()) #txt

        # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, [qcont] 查询上下文, 块长度，标签]

        if lang_type=='python':

            parse_acont1, parse_acont2,parse_query, parse_code,qids  = parse_python(corpus_lis,split_num)
            for i in range(0,len(qids)):
                total_data.append([qids[i],[parse_acont1[i],parse_acont2[i]],[parse_code[i]],parse_query[i]])

        if lang_type == 'sql':

            parse_acont1,parse_acont2,parse_query, parse_code,qids = parse_sqlang(corpus_lis, split_num)
            for i in range(0,len(qids)):
                total_data.append([qids[i],[parse_acont1[i],parse_acont2[i]],[parse_code[i]],parse_query[i]])


    f = open(save_path, "w")
    f.write(str(total_data))
    f.close()




"""
@variable: python_type
    @description: Python语言类型的标识符
    @type: str

@variable: sqlang_type
    @description: SQL语言类型的标识符
    @type: str

@variable: words_top
    @description: 词频统计的前N个词的数量
    @type: int

@variable: split_num
    @description: 数据分割数，用于将数据列表分割成较小的子列表
    @type: int

@method: test
    @description: 测试函数，用于加载和打印指定路径下的数据文件
    @package: getStructureToVector
    @reference:
        @method: main
            @description: 主函数，用于解析给定的语言类型的数据文件，并将解析结果存储到指定的文件中
            @:argument lang_type (str): 语言类型，可选值为'python'和'sql'
            @:argument split_num (int): 分割数，用于将数据列表分割成较小的子列表
            @:argument source_path (str): 源数据文件的路径
            @:argument save_path (str): 存储结果的文件路径
            @return: None 无返回值
    @param path1 (str): 第一个数据文件的路径
    @param path2 (str): 第二个数据文件的路径
    @return: None 无返回值
    @analysis:
        @original: 代码逻辑
        @explanation:
        打开第一个数据文件，加载数据到corpus_lis1
        打开第二个数据文件，加载数据到corpus_lis2
        打印corpus_lis1中索引为10的数据
        打印corpus_lis2中索引为10的数据
    @changes: 无
        
    @condition: name == 'main'
        @description: 主程序入口，判断是否为主模块执行
        @analysis:
            @original: 代码逻辑
            @explanation:
            定义staqc_python_path为Python数据文件的路径
            定义staqc_python_save为Python解析结果存储文件的路径
            定义staqc_sql_path为SQL数据文件的路径
            定义staqc_sql_save为SQL解析结果存储文件的路径
            定义large_python_path为大型Python数据文件的路径
            定义large_python_save为大型Python解析结果存储文件的路径
            定义large_sql_path为大型SQL数据文件的路径
            定义large_sql_save为大型SQL解析结果存储文件的路径
            调用main函数对staqc_python_path和staqc_python_save进行解析
            调用main函数对large_python_path和large_python_save进行解析
        @changes: 无

"""
python_type= 'python'
sqlang_type ='sql'
words_top = 100
split_num = 1000
def test(path1,path2):
    with open(path1, "rb") as f:
        #  存储为字典 有序
        corpus_lis1  = pickle.load(f) #pickle
    with open(path2, "rb") as f:
        corpus_lis2 = eval(f.read()) #txt

    print(corpus_lis1[10])
    print(corpus_lis2[10])
if __name__ == '__main__':
    staqc_python_path = '../hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save ='../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'

    staqc_sql_path = '../hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'

    #main(sqlang_type,split_num,staqc_sql_path,staqc_sql_save)
    #main(python_type, split_num, staqc_python_path, staqc_python_save)

    large_python_path='../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save='../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'


    large_sql_path='../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save='../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    #main(sqlang_type, split_num, large_sql_path, large_sql_save)
    main(python_type, split_num, large_python_path, large_python_save)