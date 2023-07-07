'''
从大词典中获取特定于于语料的词典
将数据处理成待打标签的形式
'''

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from gensim.models import KeyedVectors


'''
@method: trans_bin
    @description: 把词向量文件从文本格式转换为二进制格式的bin文件，使用gensim中的KeyedVectors来处理词向量
    @package: embddings_process.py
    @references:
        gensim.models.KeyedVectors
            @method: load_word2vec_format(path1,binary= False)
                @package: gensim.models
                @:argument: path1  原词向量文件路径
                @:argument: binary 启用二进制模式
                    
    @param: path1:原词向量文件的路径
    @param: path2:转换后的目标文件路径（二进制格式）
    @return: None 无返回值
    
    @analysis:
        @original:读取用一下代码 model = KeyedVectors.load(embed_path, mmap='r')
            @explanation: 应该是一个示例代码，告诉读者如何加载之前保存的词向量模型
            @code: 使用‘r'只读模式读取路径文件’embed_path'
        @original:如果每次都用上面的方法加载，速度非常慢，可以将词向量文件保存成bin文件，以后就加载bin文件，速度会变快
            @explanation: 应该是一个"{优化建议}"，解释为什么将词向量文件保存为二进制（bin）文件。
'''
def trans_bin(path1,path2):
    wv_from_text = KeyedVectors.load_word2vec_format(path1, binary=False)
    #如果每次都用上面的方法加载，速度非常慢，可以将词向量文件保存成bin文件，以后就加载bin文件，速度会变快
    wv_from_text.init_sims(replace=True)
    wv_from_text.save(path2)

    return

'''
    @method: get_new_dict
        @description: 构建新的词典和词向量矩阵。
        @package: embddings_process.py
        @references:
            gensim.models.KeyedVectors
                @method:load(type_vec_path, mmap='r')
                @package: gensim.models
                @:argument: type_vec_path 类型词向量文件的路径
                @:argument: mmap 读取的方式
        
        @param type_vec_path: 类型词向量文件的路径
        @param type_word_path: 类型词文件的路径
        @param final_vec_path: 构建后的词向量矩阵保存路径
        @param final_word_path: 构建后的词典保存路径
        @return: None 无返回值
    @changes
        @statement:
            @before print("完成“)
            @after  print("Method get_new_dict Finished：构建新的词典 和词向量矩阵 完成")
    """
'''
#构建新的词典 和词向量矩阵
def get_new_dict(type_vec_path,type_word_path,final_vec_path,final_word_path):  #词标签，词向量
    #原词159018 找到的词133959 找不到的词25059
    #添加unk过后 159019 找到的词133960 找不到的词25059
    #添加pad过后 词典：133961 词向量 133961
    # 加载转换文件
    model = KeyedVectors.load(type_vec_path, mmap='r')

    with open(type_word_path,'r')as f:
        total_word= eval(f.read())
        f.close()

    # 输出词向量
    word_dict = ['PAD','SOS','EOS','UNK']#其中0 PAD_ID,1SOS_ID,2E0S_ID,3UNK_ID

    fail_word = []
    rng = np.random.RandomState(None)
    pad_embedding = np.zeros(shape=(1, 300)).squeeze()
    unk_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    sos_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    eos_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    word_vectors = [pad_embedding,sos_embediing,eos_embediing,unk_embediing]
    print(len(total_word))
    for word in total_word:
        try:
            word_vectors.append(model.wv[word]) #加载词向量
            word_dict.append(word)
        except:
            print(word)
            fail_word.append(word)
    #关于有多少个词，以及多少个词没有找到
    print(len(word_dict))
    print(len(word_vectors))
    print(len(fail_word))

    #判断词向量是否正确
    '''
    couunt = 0
    for i in range(4,len(word_dict)):
        if word_vectors[i].all() == model.wv[word_dict[i]].all():
            continue
        else:
            couunt +=1

    print(couunt)
    '''
    word_vectors = np.array(word_vectors)
    #print(word_vectors.shape)
    word_dict = dict(map(reversed, enumerate(word_dict)))
    #np.savetxt(final_vec_path,word_vectors)
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)

    v = pickle.load(open(final_vec_path, 'rb'), encoding='iso-8859-1')
    with open(final_word_path, 'rb') as f:
        word_dict = pickle.load(f)
    couunt = 0

    print("Method get_new_dict Finished：构建新的词典 和词向量矩阵 完成")


'''
@method: get_index
    @description: 获取词在词典中的位置
    @package: embddingings_process.py
    @param: type: 类型，表示文本类型或代码类型
    @param: text: 输入的文本或代码
    @param: word_dict: 词典，存储词及其对应的索引位置
    @return: location，词在词典中的位置列表
    @analysis: 
        @original: 大量的条件判断语句
        @explanation:
            - 如果type为'code'，表示处理代码类型
                - 将1添加到location列表中（表示代码类型）
                - 计算text的长度，存储在len_c中
                - 如果len_c + 1 < 350，表示文本长度小于350
                    - 如果len_c为1且text[0]为'-1000'，将2添加到location列表中
                    - 否则，对于文本中的每个词
                        - 如果word_dict中存在该词，获取其索引值并添加到location列表中
                        - 否则，将'UNK'对应的索引值添加到location列表中
                    - 将2添加到location列表中（表示文本结束）
                - 否则，文本长度大于等于350
                    - 对于文本中的前348个词
                        - 如果word_dict中存在该词，获取其索引值并添加到location列表中
                        - 否则，将'UNK'对应的索引值添加到location列表中
                    - 将2添加到location列表中（表示文本结束）
            - 否则，表示处理文本类型
                - 如果文本长度为0，将0添加到location列表中
                - 否则，如果text的第一个词为'-10000'，将0添加到location列表中
                - 否则，对于文本中的每个词
                    - 如果word_dict中存在该词，获取其索引值并添加到location列表中
                    - 否则，将'UNK'对应的索引值添加到location列表中
    @changes:
        @valiableName
            @before: len_c
            @after:  text_len
'''
#得到词在词典中的位置
def get_index(type,text,word_dict):
    location = []
    if type == 'code':
        location.append(1)
        text_len = len(text)
        if text_len+1 <350:
            if text_len == 1 and text[0] == '-1000':
                location.append(2)
            else:
                for i in range(0, text_len):
                    if word_dict.get(text[i]) != None:
                        index = word_dict.get(text[i])
                        location.append(index)
                    else:
                        index = word_dict.get('UNK')
                        location.append(index)

                location.append(2)
        else:
            for i in range(0, 348):
                if word_dict.get(text[i]) != None:
                    index = word_dict.get(text[i])
                    location.append(index)
                else:
                    index = word_dict.get('UNK')
                    location.append(index)
            location.append(2)
    else:
        if len(text) == 0:
            location.append(0)
        elif text[0] == '-10000':
            location.append(0)
        else:
            for i in range(0, len(text)):
                if word_dict.get(text[i]) != None:
                    index = word_dict.get(text[i])
                    location.append(index)
                else:
                    index = word_dict.get('UNK')
                    location.append(index)

    return location



'''
    @method: Serialization
    @description: 将训练、测试、验证语料序列化
    @package: embddings_process.py
    @param: word_dict_path: 词典文件路径，包含词和对应索引的映射关系
    @param: type_path: 类型文件路径，包含待处理的语料数据
    @param: final_type_path: 序列化后的语料文件保存路径
    @return: None，无返回值
    @analysis:
        @original: 方法的所有逻辑
        @explanation:
            - 读取词典文件，加载词和对应索引的映射关系
            - 读取待处理的语料数据
            - 初始化总数据列表
            - 对于每个语料数据
                - 提取qid（查询ID）
                - 调用get_index方法，获取Si_word_list（Si词列表）：类型为'text'，文本为corpus[i][1][0]
                - 调用get_index方法，获取Si1_word_list（Si+1词列表）：类型为'text'，文本为corpus[i][1][1]
                - 调用get_index方法，获取tokenized_code（代码词列表）：类型为'code'，代码为corpus[i][2][0]
                - 调用get_index方法，获取query_word_list（查询词列表）：类型为'text'，文本为corpus[i][3]
                - 初始化block_length为4
                - 初始化label为0
                - 如果Si_word_list的长度大于100，截取前100个词，否则在末尾补0直到长度为100
                - 如果Si1_word_list的长度大于100，截取前100个词，否则在末尾补0直到长度为100
                - 如果tokenized_code的长度小于350，在末尾补0直到长度为350，否则截取前350个词
                - 如果query_word_list的长度大于25，截取前25个词，否则在末尾补0直到长度为25
                - 构建一条数据记录one_data，包括qid、[Si_word_list, Si1_word_list]、[tokenized_code]、query_word_list、block_length、label
                - 将one_data添加到总数据列表total_data中
            - 将total_data以二进制格式保存到final_type_path文件中
'''
#将训练、测试、验证语料序列化
#查询：25 上下文：100 代码：350
def Serialization(word_dict_path,type_path,final_type_path):

    with open(word_dict_path, 'rb') as f:
        word_dict = pickle.load(f)

    with open(type_path,'r')as f:
        corpus= eval(f.read())
        f.close()

    total_data = []


    for i in range(0, len(corpus)):
        qid = corpus[i][0]

        Si_word_list = get_index('text',corpus[i][1][0],word_dict)
        # Si+1
        Si1_word_list = get_index('text',corpus[i][1][1],word_dict)

        # code
        #tokenized_code = get_index('code',corpus[i][2][0],word_dict) #训练语料
        tokenized_code = get_index('code', corpus[i][2][0], word_dict) #staqc
        # query
        query_word_list = get_index('text',corpus[i][3],word_dict)
        #block_length = corpus[i][4]
        #label = corpus[i][5]
        block_length = 4
        label = 0
        if(len(Si_word_list)>100):
            Si_word_list = Si_word_list[:100]
        else:
            for k in range(0, 100 - len(Si_word_list)):
                Si_word_list.append(0)

        if (len(Si1_word_list) > 100):
            Si1_word_list = Si1_word_list[:100]
        else:
            for k in range(0, 100 - len(Si1_word_list)):
                Si1_word_list.append(0)

        if (len(tokenized_code) < 350):
            for k in range(0, 350 - len(tokenized_code)):
                tokenized_code.append(0)
        else:
            tokenized_code = tokenized_code[:350]

        if (len(query_word_list) > 25):
            query_word_list = query_word_list[:25]
        else:
            for k in range(0, 25 - len(query_word_list)):
                query_word_list.append(0)

        one_data = [qid, [Si_word_list, Si1_word_list], [tokenized_code], query_word_list, block_length, label]
        total_data.append(one_data)

    with open(final_type_path, 'wb') as file:
        pickle.dump(total_data, file)

'''
@method:get_new_dict_append
    @description:将一个词向量文件和额外的词列表添加到原词典和词向量矩阵中。
    @param type_vec_path: 原词向量文件的路径
    @param previous_dict: 原词典文件的路径
    @param previous_vec: 原词向量矩阵文件的路径
    @param append_word_path: 需要添加的词列表文件的路径
    @param final_vec_path: 转换后的词向量矩阵文件的路径
    @param final_word_path: 转换后的词典文件的路径
    @return: None 无返回值
    @analysis:
        @original:代码逻辑
        @explanation:
            加载原词向量文件，使用gensim中的KeyedVectors模块的load函数加载
            加载原词典文件和原词向量矩阵文件
            读取额外的词列表文件，存储为append_word
            打印原词向量矩阵的类型和词典的长度
            将原词向量矩阵转换为列表形式
            创建一个空的失败词列表fail_word
            随机数生成器rng，用于生成UNK词的词向量
            遍历额外的词列表append_word，将词向量添加到原词向量矩阵和词典中
            打印词典的长度、词向量矩阵的长度以及失败词的数量
            将词向量矩阵和词典转换为NumPy数组和字典形式
            将转换后的词向量矩阵和词典保存到文件中
            打印"完成"            
    
    @changes
        @statement 
            @before print("完成")
            @after  print("Method get_new_dict_append Finish:将一个词向量文件和额外的词列表添加到原词典和词向量矩阵中 完成！")
'''
def get_new_dict_append(type_vec_path,previous_dict,previous_vec,append_word_path,final_vec_path,final_word_path):  #词标签，词向量
    #原词159018 找到的词133959 找不到的词25059
    #添加unk过后 159019 找到的词133960 找不到的词25059
    #添加pad过后 词典：133961 词向量 133961
    # 加载转换文件

    model = KeyedVectors.load(type_vec_path, mmap='r')

    with open(previous_dict, 'rb') as f:
        pre_word_dict = pickle.load(f)

    with open(previous_vec, 'rb') as f:
        pre_word_vec = pickle.load(f)

    with open(append_word_path,'r')as f:
        append_word= eval(f.read())
        f.close()

    # 输出词向量

    print(type(pre_word_vec))
    word_dict =  list(pre_word_dict.keys()) #'#其中0 PAD_ID,1SOS_ID,2E0S_ID,3UNK_ID
    print(len(word_dict))
    word_vectors = pre_word_vec.tolist()
    print(word_dict[:100])
    fail_word =[]
    print(len(append_word))
    rng = np.random.RandomState(None)
    unk_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    h = []

    for word in append_word:
        try:

            word_vectors.append(model.wv[word]) #加载词向量
            word_dict.append(word)
        except:
            fail_word.append(word)
    #关于有多少个词，以及多少个词没有找到
    print(len(word_dict))
    print(len(word_vectors))
    print(len(fail_word))
    print(word_dict[:100])



    '''
    #判断词向量是否正确
    print("----------------------------")
    couunt = 0

    import operator
    for i in range(159035,len(word_dict)):
        if operator.eq(word_vectors[i].tolist(), model.wv[word_dict[i]].tolist()) == True:
            continue
        else:
            couunt +=1

    print(couunt)
    '''
    word_vectors = np.array(word_vectors)
    #print(word_vectors.shape)
    word_dict = dict(map(reversed, enumerate(word_dict)))
    #np.savetxt(final_vec_path,word_vectors)
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)


    print("Method get_new_dict_append Finish:将一个词向量文件和额外的词列表添加到原词典和词向量矩阵中 完成！")




import time

#-------------------------参数配置----------------------------------
#python 词典 ：1121543 300
if __name__ == '__main__':

    ps_path = '../hnn_process/embeddings/10_10/python_struc2vec1/data/python_struc2vec.txt' #239s
    ps_path_bin = '../hnn_process/embeddings/10_10/python_struc2vec.bin' #2s

    sql_path = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.txt'
    sql_path_bin = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin'

    #trans_bin(sql_path,sql_path_bin)
    #trans_bin(ps_path, ps_path_bin)
    #113440 27970(2) 49409(12),50226(30),55993(98)

    #==========================  ==========最初基于Staqc的词典和词向量==========================

    python_word_path = '../hnn_process/data/word_dict/python_word_vocab_dict.txt'
    python_word_vec_path = '../hnn_process/embeddings/python/python_word_vocab_final.pkl'
    python_word_dict_path = '../hnn_process/embeddings/python/python_word_dict_final.pkl'

    sql_word_path = '../hnn_process/data/word_dict/sql_word_vocab_dict.txt'
    sql_word_vec_path = '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl'
    sql_word_dict_path = '../hnn_process/embeddings/sql/sql_word_dict_final.pkl'



    #txt存储数组向量，读取时间：30s,以pickle文件存储0.23s,所以最后采用pkl文件

    #get_new_dict(ps_path_bin,python_word_path,python_word_vec_path,python_word_dict_path)
    #get_new_dict(sql_path_bin, sql_word_path, sql_word_vec_path, sql_word_dict_path)

    # =======================================最后打标签的语料========================================
    #sql 待处理语料地址
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    #sql大语料最后的词典
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'

    # sql最后的词典和对应的词向量
    sql_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl'
    sql_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'
    #get_new_dict(sql_path_bin, final_word_dict_sql, sql_final_word_vec_path, sql_final_word_dict_path)
    #get_new_dict_append(sql_path_bin, sql_word_dict_path, sql_word_vec_path, large_word_dict_sql, sql_final_word_vec_path,sql_final_word_dict_path)

    staqc_sql_f = '../hnn_process/ulabel_data/staqc/seri_sql_staqc_unlabled_data.pkl'
    large_sql_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_ql_large_multiple_unlable.pkl'
    #Serialization(sql_final_word_dict_path, new_sql_staqc, staqc_sql_f)
    #Serialization(sql_final_word_dict_path, new_sql_large, large_sql_f)



    #python
    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    final_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'

    #python最后的词典和对应的词向量
    python_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'
    python_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/python_word_dict_final.pkl'

    #get_new_dict(ps_path_bin, final_word_dict_python, python_final_word_vec_path, python_final_word_dict_path)
    #get_new_dict_append(ps_path_bin, python_word_dict_path, python_word_vec_path, large_word_dict_python, python_final_word_vec_path,python_final_word_dict_path)

    #处理成打标签的形式
    staqc_python_f ='../hnn_process/ulabel_data/staqc/seri_python_staqc_unlabled_data.pkl'
    large_python_f ='../hnn_process/ulabel_data/large_corpus/multiple/seri_python_large_multiple_unlable.pkl'
    #Serialization(python_final_word_dict_path, new_python_staqc, staqc_python_f)
    Serialization(python_final_word_dict_path, new_python_large, large_python_f)

    print('序列化完毕')
    #test2(test_python1,test_python2,python_final_word_dict_path,python_final_word_vec_path)








