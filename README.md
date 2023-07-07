# hnn_process
软件工程实习，代码修改

# 软件工程代码规范化作业

有一项目，hnn_process，其中的代码不符合规范，难以读懂，请你通过修改以增加代码的可读性。

## 文件树结构

.<br>
│  .gitattributes<br>
│  embddings_process.py<br>
│  getStrucutureToVector.py<br>
│  process_single_corpus.py<br>
│  python_structured.py<br>
│  sqlang_structured.py<br>
│  word_dict.py<br>

## 修改内容

为原有代码进行了一部分修改，并且对诸多难以理解功能的函数新增了注释。

注释规范如下：

@method: 表示要注释的方法<br>
@description: 方法功能的描述<br>
[@reference: 方法引用的库内容]<br>
[@package: 方法所在文件名]<br>
@param: 方法的形式参数描述<br>
@return: 方法的返回值描述<br>
@analysis: 对原有代码的注释、方法变量、业务逻辑等内容的分析。<br>
@original: 描述原有内容<br>
@explanation: 解释original的内容<br>
[@changes: 对代码内容的修改]<br>

## 修改示例

<pre><code>
@method: trans_bin
    @description: 把词向量文件从文本格式转换为二进制格式的bin文件，使用gensim中的KeyedVectors来处理词向量
    @package: embddings_process.py
    @references:
        gensim.models.KeyedVectors
            @method: load_word2vec_format(path1,binary= False)
                @package: gensim.models
                @argument: path1  原词向量文件路径
                @argument: binary 启用二进制模式
                    
    @param: path1: 原词向量文件的路径
    @param: path2: 转换后的目标文件路径（二进制格式）
    @return: None 无返回值
    
    @analysis:
        @original: 读取以下代码 model = KeyedVectors.load(embed_path, mmap='r')
            @explanation: 应该是一个示例代码，告诉读者如何加载之前保存的词向量模型
            @code: 使用 'r' 只读模式读取路径文件 'embed_path'
        @original: 如果每次都用上面的方法加载，速度非常慢，可以将词向量文件保存成bin文件，以后就加载bin文件，速度会变快
            @explanation: 应该是一个"{优化建议}"，解释为什么将词向量文件保存为二进制（bin）文件。
</code>
</pre>
下面是源代码
<pre><code>
def trans_bin(path1, path2):
    wv_from_text = KeyedVectors.load_word2vec_format(path1, binary=False)
    # 如果每次都用上面的方法加载，速度非常慢，可以将词向量文件保存成bin文件，以后就加载bin文件，速度会变快
    wv_from_text.init_sims(replace=True)
    wv_from_text.save(path2)

    return
</code>
</pre>
