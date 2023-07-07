# -*- coding: utf-8 -*-
import re
import ast
import sys
import token
import tokenize

from nltk import wordpunct_tokenize
from io import StringIO
# 骆驼命名法
import inflection

# 词性还原
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
wnler = WordNetLemmatizer()

# 词干提取
from nltk.corpus import wordnet

#############################################################################

PATTERN_VAR_EQUAL = re.compile("(\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)(,\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)*=")
PATTERN_VAR_FOR = re.compile("for\s+[_a-zA-Z][_a-zA-Z0-9]*\s*(,\s*[_a-zA-Z][_a-zA-Z0-9]*)*\s+in")

"""
@method: repair_program_io
    @description: 修复编程代码中的输入输出格式，将代码修复为规范的格式，并将修复后的代码和代码列表返回。
    @package：python_structured.py
    @param code (str): 待修复的编程代码。
    @return: repaired_code (str): 修复后的代码。
            code_list (list): 修复后的代码列表。
    @analysis:
        @original: 代码逻辑
        @explanation:
            定义正则表达式模式用于匹配不同的情况
            将代码按行分割为列表lines
            创建一个与lines长度相同的列表lines_flags，并初始化为0
            创建一个空的代码列表code_list
            遍历每行代码，根据模式匹配进行标记，将标记结果存入lines_flags
            将lines_flags转换为字符串lines_flags_string
            若lines_flags中没有0，则无需修复，直接将修复后的代码设置为原代码，code_list为原代码的列表形式，并标记修复完成
            若lines_flags_string符合特定模式（连续出现的1和3或连续出现的4和5），则需要进行修复
            初始化修复后的代码repaired_code为空字符串
            初始化前一个标记的索引pre_idx为0
            初始化子块代码sub_block为空字符串
            若第一个标记为0，则在repaired_code中加入lines[pre_idx]，并将pre_idx和flag更新为下一个非0的标记
            将sub_block设置为repaired_code的值，并将其添加到code_list中，然后清空sub_block
            遍历lines_flags从pre_idx到末尾
                若当前标记不为0，则将去除标记后的行添加到repaired_code中，并根据前一个行的标记判断是否将sub_block添加到code_list中
                若当前标记为0，则根据前一个行的标记判断是否将sub_block添加到code_list中，并将当前行添加到sub_block中
            避免遗漏最后一个单元，若sub_block非空，则将其添加到code_list中
            若修复后的代码repaired_code非空，则标记修复完成
            若不符合上述情况，则属于非典型情况，只需在每个Out后移除0标记的行
                初始化修复后的代码repaired_code为空字符串
                初始化子块代码sub_block为空字符串
                初始化bool_after_Out为False
                遍历lines_flags
                    若当前标记不为0，则根据不同的标记操作修复后的代码和sub_block，并更新bool_after_Out的值
                    若当前标记为0且bool_after_Out为False，则在repaired_code中加入当前行
    
                    若当前标记为0且前一个行的标记不为0，则将sub_block添加到code_list中，并清空sub_block
                    将当前行添加到sub_block中

@changes: 无
"""
def repair_program_io(code):

    # reg patterns for case 1
    pattern_case1_in = re.compile("In ?\[\d+\]: ?")  # flag1
    pattern_case1_out = re.compile("Out ?\[\d+\]: ?")  # flag2
    pattern_case1_cont = re.compile("( )+\.+: ?")  # flag3

    # reg patterns for case 2
    pattern_case2_in = re.compile(">>> ?")  # flag4
    pattern_case2_cont = re.compile("\.\.\. ?")  # flag5

    patterns = [pattern_case1_in, pattern_case1_out, pattern_case1_cont,
                pattern_case2_in, pattern_case2_cont]

    lines = code.split("\n")
    lines_flags = [0 for _ in range(len(lines))]

    code_list = []  # a list of strings

    # match patterns
    for line_idx in range(len(lines)):
        line = lines[line_idx]
        for pattern_idx in range(len(patterns)):
            if re.match(patterns[pattern_idx], line):
                lines_flags[line_idx] = pattern_idx + 1
                break
    lines_flags_string = "".join(map(str, lines_flags))

    bool_repaired = False

    # pdb.set_trace()
    # repair
    if lines_flags.count(0) == len(lines_flags):  # no need to repair
        repaired_code = code
        code_list = [code]
        bool_repaired = True

    elif re.match(re.compile("(0*1+3*2*0*)+"), lines_flags_string) or \
            re.match(re.compile("(0*4+5*0*)+"), lines_flags_string):
        repaired_code = ""
        pre_idx = 0
        sub_block = ""
        if lines_flags[0] == 0:
            flag = 0
            while (flag == 0):
                repaired_code += lines[pre_idx] + "\n"
                pre_idx += 1
                flag = lines_flags[pre_idx]
            sub_block = repaired_code
            code_list.append(sub_block.strip())
            sub_block = ""  # clean

        for idx in range(pre_idx, len(lines_flags)):
            if lines_flags[idx] != 0:
                repaired_code += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

                # clean sub_block record
                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] == 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

            else:
                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] != 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += lines[idx] + "\n"

        # avoid missing the last unit
        if len(sub_block.strip()):
            code_list.append(sub_block.strip())

        if len(repaired_code.strip()) != 0:
            bool_repaired = True

    if not bool_repaired:  # not typical, then remove only the 0-flag lines after each Out.
        repaired_code = ""
        sub_block = ""
        bool_after_Out = False
        for idx in range(len(lines_flags)):
            if lines_flags[idx] != 0:
                if lines_flags[idx] == 2:
                    bool_after_Out = True
                else:
                    bool_after_Out = False
                repaired_code += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] == 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

            else:
                if not bool_after_Out:
                    repaired_code += lines[idx] + "\n"

                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] != 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += lines[idx] + "\n"

    return repaired_code, code_list


'''
@method: get_vars
    @description: 获取抽象语法树(ast)中的变量列表。
    @param ast_root (ast.AST): 抽象语法树的根节点。
    @return: vars (list): 排序后的变量列表。
    @analysis:
        @original: 代码逻辑
        @explanation:
            使用列表推导式，遍历抽象语法树(ast)中的每个节点
            对于每个节点，判断是否为ast.Name类型且上下文不是ast.Load
            如果满足条件，将节点的id添加到变量集合中

            返回排序后的变量列表
    @changes: 无
'''
def get_vars(ast_root):
    return sorted(
        {node.id for node in ast.walk(ast_root) if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)})

'''
@method: get_vars_heuristics
    @description: 根据启发式规则从代码中获取变量名集合。
    @param code (str): 代码字符串。
    @return: varnames (set): 变量名集合。
    @analysis:
        @original: 代码逻辑
        @explanation:
            将代码按行分割，并过滤掉空行，得到代码行列表code_lines

            最佳尝试解析：
            初始化start为0，end为代码行列表的长度减1，bool_success为False
            当bool_success为False时，进行如下循环：
                尝试使用ast.parse解析code_lines[start:end]连接后的字符串，如果出现异常则end减1，直到解析成功
            将解析得到的根节点root传递给get_vars函数，并将返回的变量名集合与varnames合并

            处理剩余的代码行：
            对于每一行代码line：
                去除首尾空格并赋值给line
                尝试使用ast.parse解析line，如果出现异常则执行下面的匹配逻辑：
                    匹配PATTERN_VAR_EQUAL：
                        使用re.match匹配PATTERN_VAR_EQUAL和line，如果匹配成功则执行以下操作：
                            将匹配结果去除最后的"="并赋值给match
                            使用逗号分割match，并去除空格后添加到varnames中
                    匹配PATTERN_VAR_FOR：
                        使用re.search匹配PATTERN_VAR_FOR和line，如果匹配成功则执行以下操作：
                            将匹配结果去除开头的"for"和结尾的"in"并赋值给match
                            使用逗号分割match，并去除空格后添加到varnames中

                如果解析成功，将返回的变量名集合与varnames合并

            返回变量名集合varnames
    @changes: 无
'''
def get_vars_heuristics(code):
    varnames = set()
    code_lines = [_ for _ in code.split("\n") if len(_.strip())]

    # best effort parsing
    start = 0
    end = len(code_lines) - 1
    bool_success = False
    while (not bool_success):
        try:
            root = ast.parse("\n".join(code_lines[start:end]))
        except:
            end -= 1
        else:
            bool_success = True
    # print("Best effort parse at: start = %d and end = %d." % (start, end))
    varnames = varnames.union(set(get_vars(root)))
    # print("Var names from base effort parsing: %s." % str(varnames))

    # processing the remaining...
    for line in code_lines[end:]:
        line = line.strip()
        try:
            root = ast.parse(line)
        except:
            # matching PATTERN_VAR_EQUAL
            pattern_var_equal_matched = re.match(PATTERN_VAR_EQUAL, line)
            if pattern_var_equal_matched:
                match = pattern_var_equal_matched.group()[:-1]  # remove "="
                varnames = varnames.union(set([_.strip() for _ in match.split(",")]))

            # matching PATTERN_VAR_FOR
            pattern_var_for_matched = re.search(PATTERN_VAR_FOR, line)
            if pattern_var_for_matched:
                match = pattern_var_for_matched.group()[3:-2]  # remove "for" and "in"
                varnames = varnames.union(set([_.strip() for _ in match.split(",")]))

        else:
            varnames = varnames.union(get_vars(root))

    return varnames

'''
@method: PythonParser
    @package: python_structured.py
    @description: 解析给定的Python代码并返回标记化的代码和解析状态
    @param code: 要解析的Python代码字符串
    @return: 
        - tokenized_code: 标记化后的代码列表
        - bool_failed_var: 变量解析是否失败的布尔值
        - bool_failed_token: 标记解析是否失败的布尔值
    
    @algorithm:
        - 尝试将代码解析为抽象语法树，获取其中的变量名集合
        - 若解析失败，则修复程序输入输出格式，再次尝试解析
        - 若仍然解析失败，则使用启发式方法获取变量名集合
        - 初始化空的标记化代码列表
        - 定义函数first_trial，用于第一次尝试标记化代码的处理
        - 进行第一次尝试，若失败则逐步去除开头的字符直至成功
        - 使用tokenize模块生成代码的标记流
        - 遍历标记流中的每个标记
            - 若标记为数字、字符串或换行符，将其名称添加到标记化代码列表中
            - 若标记不是注释或文件结束符，并且具有非空内容
                - 若该内容不在变量名集合中，将其添加到标记化代码列表中
                - 否则，将"VAR"添加到标记化代码列表中
            - 获取下一个标记，直至遍历完所有标记或出现异常
        - 返回标记化代码列表和解析状态的布尔值
    @changes: 修改了一部分变量名，新增了一些注释
'''
def PythonParser(code):
    bool_failed_var = False  # 标记变量解析是否失败
    bool_failed_token = False  # 标记标记解析是否失败

    try:
        root = ast.parse(code)  # 尝试将代码解析为抽象语法树
        varnames = set(get_vars(root))  # 获取变量名集合
    except:
        repaired_code, _ = repair_program_io(code)  # 修复程序输入输出格式
        try:
            root = ast.parse(repaired_code)  # 尝试将修复后的代码解析为抽象语法树
            varnames = set(get_vars(root))  # 获取变量名集合
        except:
            bool_failed_var = True  # 变量解析失败
            varnames = get_vars_heuristics(code)  # 使用启发式方法获取变量名集合

    tokenized_code = []  # 存储标记化后的代码

    def first_trial(_code):
        # 第一次尝试标记化代码的函数

        if len(_code) == 0:
            return True
        try:
            g = tokenize.generate_tokens(StringIO(_code).readline)
            term = next(g)
        except:
            return False
        else:
            return True

    bool_first_success = first_trial(code)
    while not bool_first_success:
        code = code[1:]
        bool_first_success = first_trial(code)

    g = tokenize.generate_tokens(StringIO(code).readline)
    term = next(g)

    bool_finished = False
    while not bool_finished:
        term_type = term[0]
        lineno = term[2][0] - 1
        posno = term[3][1] - 1
        if token.tok_name[term_type] in {"NUMBER", "STRING", "NEWLINE"}:
            tokenized_code.append(token.tok_name[term_type])
        elif (
            not token.tok_name[term_type] in {"COMMENT", "ENDMARKER"}
            and len(term[1].strip())
        ):
            candidate = term[1].strip()
            if candidate not in varnames:
                tokenized_code.append(candidate)
            else:
                tokenized_code.append("VAR")

        # 获取下一个标记
        bool_success_next = False
        while not bool_success_next:
            try:
                term = next(g)
            except StopIteration:
                bool_finished = True
                break
            except:
                bool_failed_token = True  # 标记标记解析失败
                code_lines = code.split("\n")
                if lineno > len(code_lines) - 1:
                    print(sys.exc_info())
                else:
                    failed_code_line = code_lines[lineno]  # 错误的代码行
                    if posno < len(failed_code_line) - 1:
                        failed_code_line = failed_code_line[posno:]
                        tokenized_failed_code_line = wordpunct_tokenize(
                            failed_code_line
                        )  # 对错误的代码行进行标记化处理
                        tokenized_code += tokenized_failed_code_line
                    if lineno < len(code_lines) - 1:
                        code = "\n".join(code_lines[lineno + 1:])
                        g = tokenize.generate_tokens(StringIO(code).readline)
                    else:
                        bool_finished = True
                        break
            else:
                bool_success_next = True

    return tokenized_code, bool_failed_var, bool_failed_token


'''
@method: revert_abbrev
    @package: python_structured.py
    @description: 将缩写还原为完整形式的文本
    @param line: 要还原的文本行
    @return: 还原后的文本行
    @algorithm:
        - 使用正则表达式匹配缩写形式，并进行替换
        - 将特定的缩写形式替换为对应的完整形式
        - 返回还原后的文本行'''
#############################################################################

#############################################################################

# 缩略词处理
def revert_abbrev(line):
    pat_is = re.compile("(it|he|she|that|this|there|here)(\"s)", re.I)
    # 's
    pat_s1 = re.compile("(?<=[a-zA-Z])\"s")
    # s
    pat_s2 = re.compile("(?<=s)\"s?")
    # not
    pat_not = re.compile("(?<=[a-zA-Z])n\"t")
    # would
    pat_would = re.compile("(?<=[a-zA-Z])\"d")
    # will
    pat_will = re.compile("(?<=[a-zA-Z])\"ll")
    # am
    pat_am = re.compile("(?<=[I|i])\"m")
    # are
    pat_are = re.compile("(?<=[a-zA-Z])\"re")
    # have
    pat_ve = re.compile("(?<=[a-zA-Z])\"ve")

    line = pat_is.sub(r"\1 is", line)
    line = pat_s1.sub("", line)
    line = pat_s2.sub("", line)
    line = pat_not.sub(" not", line)
    line = pat_would.sub(" would", line)
    line = pat_will.sub(" will", line)
    line = pat_am.sub(" am", line)
    line = pat_are.sub(" are", line)
    line = pat_ve.sub(" have", line)

    return line


'''
@method: get_wordpos
    @package: python_structured.py
    @description: 获取词性标签的WordNet常量值
    @param tag: 词性标签
    @return: 对应的WordNet常量值（如形容词、动词、名词、副词），若无法匹配则返回None
    
    @algorithm:
        - 根据词性标签的首字母进行匹配
        - 若以'J'开头，返回形容词的WordNet常量值
        - 若以'V'开头，返回动词的WordNet常量值
        - 若以'N'开头，返回名词的WordNet常量值
        - 若以'R'开头，返回副词的WordNet常量值
        - 否则，返回None
'''
# 获取词性
def get_wordpos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None





# ---------------------子函数1：句子的去冗--------------------
def process_nl_line(line):
    # 句子预处理
    line = revert_abbrev(line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = line.replace('\n', ' ')
    line = re.sub(' +', ' ', line)
    line = line.strip()
    # 骆驼命名转下划线
    line = inflection.underscore(line)

    # 去除括号里内容
    space = re.compile(r"\([^\(|^\)]+\)")  # 后缀匹配
    line = re.sub(space, '', line)
    # 去除开始和末尾空格
    line=line.strip()
    return line


# ---------------------子函数1：句子的分词--------------------
def process_sent_word(line):
    # 找单词
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)
    # 替换小数
    decimal = re.compile(r"\d+(\.\d+)+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换字符串
    string = re.compile(r'\"[^\"]+\"')
    line = re.sub(string, 'TAGSTR', line)
    # 替换十六进制
    decimal = re.compile(r"0[xX][A-Fa-f0-9]+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换数字 56
    number = re.compile(r"\s?\d+\s?")
    line = re.sub(number, ' TAGINT ', line)
    # 替换字符 6c60b8e1
    other = re.compile(r"(?<![A-Z|a-z|_|])\d+[A-Za-z]+")  # 后缀匹配
    line = re.sub(other, 'TAGOER', line)
    cut_words= line.split(' ')
    # 全部小写化
    cut_words = [x.lower() for x in cut_words]
    # 词性标注
    word_tags = pos_tag(cut_words)
    tags_dict = dict(word_tags)
    word_list = []
    for word in cut_words:
        word_pos = get_wordpos(tags_dict[word])
        if word_pos in ['a', 'v', 'n', 'r']:
            # 词性还原
            word = wnler.lemmatize(word, pos=word_pos)
        # 词干提取(效果最好）
        word = wordnet.morphy(word) if wordnet.morphy(word) else word
        word_list.append(word)
    return word_list


#############################################################################

def filter_all_invachar(line):
    # 去除非常用符号；防止解析有误
    line = re.sub('[^(0-9|a-z|A-Z|\-|_|\'|\"|\-|\(|\)|\n)]+', ' ', line)
    # 包括\r\t也清除了
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line


def filter_part_invachar(line):
    #去除非常用符号；防止解析有误
    line= re.sub('[^(0-9|a-z|A-Z|\-|#|/|_|,|\'|=|>|<|\"|\-|\\|\(|\)|\?|\.|\*|\+|\[|\]|\^|\{|\}|\n)]+',' ', line)
    #包括\r\t也清除了
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line

########################主函数：代码的tokens#################################
def python_code_parse(line):
    line = filter_part_invachar(line)
    line = re.sub('\.+', '.', line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = re.sub('>>+', '', line)  # 新增加
    line = re.sub(' +', ' ', line)
    line = line.strip('\n').strip()
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    '''
    line = filter_part_invachar(line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = re.sub(' +', ' ', line)
    line = line.strip('\n').strip()
    '''
    try:
        typedCode, failed_var, failed_token  = PythonParser(line)
        # 骆驼命名转下划线
        typedCode = inflection.underscore(' '.join(typedCode)).split(' ')

        cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typedCode]
        # 全部小写化
        token_list = [x.lower()  for x in cut_tokens]
        # 列表里包含 '' 和' '
        token_list = [x.strip() for x in token_list if x.strip() != '']
        return token_list
        # 存在为空的情况，词向量要进行判断
    except:
        return  '-1000'

########################主函数：代码的tokens#################################


#######################主函数：句子的tokens##################################

def python_query_parse(line):
    line = filter_all_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    #分完词后,再去掉 括号
    for i in range(0, len(word_list)):
        if re.findall('[\(\)]', word_list[i]):
            word_list[i] = ''
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空

    return word_list


def python_context_parse(line):
    line = filter_part_invachar(line)
    #在这一步的时候驼峰命名被转换成了下划线
    line = process_nl_line(line)
    print(line)
    word_list = process_sent_word(line)
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空
    return word_list

#######################主函数：句子的tokens##################################

if __name__ == '__main__':

    print(python_query_parse("change row_height and column_width in libreoffice calc use python tagint"))
    print(python_query_parse('What is the standard way to add N seconds to datetime.time in Python?'))
    print(python_query_parse("Convert INT to VARCHAR SQL 11?"))
    print(python_query_parse('python construct a dictionary {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 0, 2], 3: [0, 0, 3], ...,999: [9, 9, 9]}'))

    print(python_context_parse('How to calculateAnd the value of the sum of squares defined as \n 1^2 + 2^2 + 3^2 + ... +n2 until a user specified sum has been reached sql()'))
    print(python_context_parse('how do i display records (containing specific) information in sql() 11?'))
    print(python_context_parse('Convert INT to VARCHAR SQL 11?'))

    print(python_code_parse('if(dr.HasRows)\n{\n // ....\n}\nelse\n{\n MessageBox.Show("ReservationAnd Number Does Not Exist","Error", MessageBoxButtons.OK, MessageBoxIcon.Asterisk);\n}'))
    print(python_code_parse('root -> 0.0 \n while root_ * root < n: \n root = root + 1 \n print(root * root)'))
    print(python_code_parse('root = 0.0 \n while root * root < n: \n print(root * root) \n root = root + 1'))
    print(python_code_parse('n = 1 \n while n <= 100: \n n = n + 1 \n if n > 10: \n  break print(n)'))
    print(python_code_parse("diayong(2) def sina_download(url, output_dir='.', merge=True, info_only=False, **kwargs):\n    if 'news.sina.com.cn/zxt' in url:\n        sina_zxt(url, output_dir=output_dir, merge=merge, info_only=info_only, **kwargs)\n  return\n\n    vid = match1(url, r'vid=(\\d+)')\n    if vid is None:\n        video_page = get_content(url)\n        vid = hd_vid = match1(video_page, r'hd_vid\\s*:\\s*\\'([^\\']+)\\'')\n  if hd_vid == '0':\n            vids = match1(video_page, r'[^\\w]vid\\s*:\\s*\\'([^\\']+)\\'').split('|')\n            vid = vids[-1]\n\n    if vid is None:\n        vid = match1(video_page, r'vid:\"?(\\d+)\"?')\n    if vid:\n   sina_download_by_vid(vid, output_dir=output_dir, merge=merge, info_only=info_only)\n    else:\n        vkey = match1(video_page, r'vkey\\s*:\\s*\"([^\"]+)\"')\n        if vkey is None:\n            vid = match1(url, r'#(\\d+)')\n            sina_download_by_vid(vid, output_dir=output_dir, merge=merge, info_only=info_only)\n            return\n        title = match1(video_page, r'title\\s*:\\s*\"([^\"]+)\"')\n        sina_download_by_vkey(vkey, title=title, output_dir=output_dir, merge=merge, info_only=info_only)"))

    print(python_code_parse("d = {'x': 1, 'y': 2, 'z': 3} \n for key in d: \n  print (key, 'corresponds to', d[key])"))
    print(python_code_parse('  #       page  hour  count\n # 0     3727441     1   2003\n # 1     3727441     2    654\n # 2     3727441     3   5434\n # 3     3727458     1    326\n # 4     3727458     2   2348\n # 5     3727458     3   4040\n # 6   3727458_1     4    374\n # 7   3727458_1     5   2917\n # 8   3727458_1     6   3937\n # 9     3735634     1   1957\n # 10    3735634     2   2398\n # 11    3735634     3   2812\n # 12    3768433     1    499\n # 13    3768433     2   4924\n # 14    3768433     3   5460\n # 15  3768433_1     4   1710\n # 16  3768433_1     5   3877\n # 17  3768433_1     6   1912\n # 18  3768433_2     7   1367\n # 19  3768433_2     8   1626\n # 20  3768433_2     9   4750\n'))
