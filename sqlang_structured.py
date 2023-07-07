# -*- coding: utf-8 -*-
import re
import sqlparse #0.4.2

#骆驼命名法
import inflection

#词性还原
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
wnler = WordNetLemmatizer()

#词干提取
from nltk.corpus import wordnet

#############################################################################
OTHER = 0
FUNCTION = 1
BLANK = 2
KEYWORD = 3
INTERNAL = 4

TABLE = 5
COLUMN = 6
INTEGER = 7
FLOAT = 8
HEX = 9
STRING = 10
WILDCARD = 11

SUBQUERY = 12

DUD = 13

ttypes = {0: "OTHER", 1: "FUNCTION", 2: "BLANK", 3: "KEYWORD", 4: "INTERNAL", 5: "TABLE", 6: "COLUMN", 7: "INTEGER",
          8: "FLOAT", 9: "HEX", 10: "STRING", 11: "WILDCARD", 12: "SUBQUERY", 13: "DUD", }

scanner = re.Scanner([(r"\[[^\]]*\]", lambda scanner, token: token), (r"\+", lambda scanner, token: "REGPLU"),
                      (r"\*", lambda scanner, token: "REGAST"), (r"%", lambda scanner, token: "REGCOL"),
                      (r"\^", lambda scanner, token: "REGSTA"), (r"\$", lambda scanner, token: "REGEND"),
                      (r"\?", lambda scanner, token: "REGQUE"),
                      (r"[\.~``;_a-zA-Z0-9\s=:\{\}\-\\]+", lambda scanner, token: "REFRE"),
                      (r'.', lambda scanner, token: None), ])

#---------------------子函数1：代码的规则--------------------
def tokenizeRegex(s):
    results = scanner.scan(s)[0]
    return results

#---------------------子函数2：代码的规则--------------------
class SqlangParser():

    """
    @method: sanitizeSql
        @package: sqlang_structured.py
        @description: 对SQL语句进行预处理，去除多余空格和符号，并做特殊字符的处理。
        @param sql (str): 待处理的SQL语句。
        @return: 处理后的SQL语句。
        @analysis:
            @original: 代码逻辑
            @explanation:
                1. 使用strip()方法去除SQL语句两端的空格，并将其转换为小写。
                2. 判断SQL语句的最后一个字符是否为分号(";")，如果不是，则在末尾添加分号。
                3. 使用re.sub()方法，将括号"("替换为" ( "，将括号")"替换为" ) "，以便将括号与其他字符分隔开。
                4. 针对特定的单词（'index', 'table', 'day', 'year', 'user', 'text'），通过正则表达式进行处理：
                   - 将单词前面的非单词字符和单词结尾的特殊字符替换为该字符加上"1"，例如：word -> word1，word- -> word1-
                   - 将单词前面的非单词字符、单词和单词后面的非单词字符替换为该字符加上"1"加上该字符，例如：-word- -> -word1-
                5. 使用replace()方法，将字符串中的"#"替换为空字符。
                6. 返回处理后的SQL语句。
        @changes: 无
    """
    @staticmethod
    def sanitizeSql(sql):
        s = sql.strip().lower()
        if not s[-1] == ";":
            s += ';'
        s = re.sub(r'\(', r' ( ', s)
        s = re.sub(r'\)', r' ) ', s)
        words = ['index', 'table', 'day', 'year', 'user', 'text']
        for word in words:
            s = re.sub(r'([^\w])' + word + '$', r'\1' + word + '1', s)
            s = re.sub(r'([^\w])' + word + r'([^\w])', r'\1' + word + '1' + r'\2', s)
        s = s.replace('#', '')
        return s


    '''
    @method: parseStrings
        @package: sqlang_structured.py
        @description: 解析SQL语句中的字符串，并进行处理。
        @param tok (sqlparse.sql.Token): 要解析的SQL语句中的一个token。
        @return: 无
        @analysis:
            @original: 代码逻辑
            @explanation:
                1. 如果tok是一个TokenList类型的对象，则遍历其中的每个子token，递归调用parseStrings()方法。
                2. 如果tok的ttype属性为STRING，说明是一个字符串类型的token。
                   - 如果self.regex为True，表示需要对字符串进行正则表达式的处理，调用tokenizeRegex()方法并将处理后的结果用空格连接起来赋值给tok.value。
                   - 如果self.regex为False，将tok.value赋值为"CODSTR"表示这是一个字符串的占位符。
        @changes: 无
    '''
    def parseStrings(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            for c in tok.tokens:
                self.parseStrings(c)
        elif tok.ttype == STRING:
            if self.regex:
                tok.value = ' '.join(tokenizeRegex(tok.value))
            else:
                tok.value = "CODSTR"


    '''
    @method: renameIdentifiers
        @package: sqlang_structured.py
        @description: 重命名标识符（列名、表名）。
        @param tok (sqlparse.sql.Token): 要重命名的标识符的token。
        @return: 无
        @analysis:
            @original: 代码逻辑
            @explanation:
                1. 如果tok是一个TokenList类型的对象，则遍历其中的每个子token，递归调用renameIdentifiers()方法。
                2. 根据tok的ttype属性进行不同的处理：
                   - 如果tok的ttype为COLUMN，表示是一个列名的token。
                     - 如果该列名在idMap["COLUMN"]中不存在，说明是一个新的列名，为其生成一个新的名称并将其添加到idMap["COLUMN"]中。
                       生成的新名称为"col" + idCount["COLUMN"]，idCount["COLUMN"]自增1。
                     - 将tok的value设置为idMap["COLUMN"][str(tok)]，即将列名替换为新的名称。
                   - 如果tok的ttype为TABLE，表示是一个表名的token。
                     - 如果该表名在idMap["TABLE"]中不存在，说明是一个新的表名，为其生成一个新的名称并将其添加到idMap["TABLE"]中。
                       生成的新名称为"tab" + idCount["TABLE"]，idCount["TABLE"]自增1。
                     - 将tok的value设置为idMap["TABLE"][str(tok)]，即将表名替换为新的名称。
                   - 如果tok的ttype为FLOAT，将tok的value设置为"CODFLO"，表示这是一个浮点数的占位符。
                   - 如果tok的ttype为INTEGER，将tok的value设置为"CODINT"，表示这是一个整数的占位符。
                   - 如果tok的ttype为HEX，将tok的value设置为"CODHEX"，表示这是一个十六进制数的占位符。
        @changes: 无
    '''
    def renameIdentifiers(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            for c in tok.tokens:
                self.renameIdentifiers(c)
        elif tok.ttype == COLUMN:
            if str(tok) not in self.idMap["COLUMN"]:
                colname = "col" + str(self.idCount["COLUMN"])
                self.idMap["COLUMN"][str(tok)] = colname
                self.idMapInv[colname] = str(tok)
                self.idCount["COLUMN"] += 1
            tok.value = self.idMap["COLUMN"][str(tok)]
        elif tok.ttype == TABLE:
            if str(tok) not in self.idMap["TABLE"]:
                tabname = "tab" + str(self.idCount["TABLE"])
                self.idMap["TABLE"][str(tok)] = tabname
                self.idMapInv[tabname] = str(tok)
                self.idCount["TABLE"] += 1
            tok.value = self.idMap["TABLE"][str(tok)]

        elif tok.ttype == FLOAT:
            tok.value = "CODFLO"
        elif tok.ttype == INTEGER:
            tok.value = "CODINT"
        elif tok.ttype == HEX:
            tok.value = "CODHEX"

    def __hash__(self):
        return hash(tuple([str(x) for x in self.tokensWithBlanks]))

    '''
    @method: __init__
    @package: sqlang_structured.py
    @description: 类的初始化方法。
    @param sql (str): 要解析的SQL语句。
    @param regex (bool): 是否使用正则表达式进行解析，默认为False。
    @param rename (bool): 是否对标识符进行重命名，默认为True。
    @return: 无
    @analysis:
        @original: 代码逻辑
        @explanation:
            1. 调用SqlangParser.sanitizeSql()方法对输入的SQL语句进行预处理，并将结果赋值给self.sql。
            2. 初始化idMap，idMapInv和idCount，分别用于存储列名和表名的映射关系及计数。
            3. 将regex参数的值赋给self.regex，表示是否使用正则表达式进行解析。
            4. 初始化parseTreeSentinel和tableStack，用于标记解析过程中的状态。
            5. 调用sqlparse.parse()方法对self.sql进行解析，并将解析结果赋给self.parse。
               注意：这里只保留解析结果中的第一个表达式。
            6. 依次调用各个辅助方法对解析结果进行处理：
               - 调用self.removeWhitespaces()方法去除解析结果中的空白符。
               - 调用self.identifyLiterals()方法识别解析结果中的字面值。
               - 将self.parse[0]的ptype属性设置为SUBQUERY，表示是一个子查询。
               - 调用self.identifySubQueries()方法识别解析结果中的子查询。
               - 调用self.identifyFunctions()方法识别解析结果中的函数。
               - 调用self.identifyTables()方法识别解析结果中的表。
               - 调用self.parseStrings()方法处理解析结果中的字符串。
            7. 如果rename为True，调用self.renameIdentifiers()方法对解析结果中的标识符进行重命名。
            8. 调用SqlangParser.getTokens()方法获取解析结果中的所有token，并将结果赋给self.tokens。
    @changes: 无

    '''
    def __init__(self, sql, regex=False, rename=True):

        self.sql = SqlangParser.sanitizeSql(sql)

        self.idMap = {"COLUMN": {}, "TABLE": {}}
        self.idMapInv = {}
        self.idCount = {"COLUMN": 0, "TABLE": 0}
        self.regex = regex

        self.parseTreeSentinel = False
        self.tableStack = []

        self.parse = sqlparse.parse(self.sql)
        self.parse = [self.parse[0]]

        self.removeWhitespaces(self.parse[0])
        self.identifyLiterals(self.parse[0])
        self.parse[0].ptype = SUBQUERY
        self.identifySubQueries(self.parse[0])
        self.identifyFunctions(self.parse[0])
        self.identifyTables(self.parse[0])

        self.parseStrings(self.parse[0])

        if rename:
            self.renameIdentifiers(self.parse[0])

        self.tokens = SqlangParser.getTokens(self.parse)


    '''
    @method: getTokens
        @package: sqlang_structured.py
        @description: 静态方法，用于从解析结果中提取所有的token。
        @param parse (list): 解析结果，包含多个表达式。
        @return: flatParse (list): 所有提取到的token列表。
        @analysis:
            @original: 代码逻辑
            @explanation:
                1. 初始化一个空列表flatParse，用于存储提取到的token。
                2. 遍历解析结果中的每个表达式：
                   - 对于每个表达式，遍历其中的每个token。
                   - 判断token的类型是否为STRING，如果是，则将字符串按空格分割，并将分割后的结果添加到flatParse中。
                   - 如果token的类型不是STRING，直接将其转换为字符串并添加到flatParse中。
                3. 返回flatParse作为结果。
        @changes: 无
    '''
    @staticmethod
    def getTokens(parse):
        flatParse = []
        for expr in parse:
            for token in expr.flatten():
                if token.ttype == STRING:
                    flatParse.extend(str(token).split(' '))
                else:
                    flatParse.append(str(token))
        return flatParse

    '''
    @method: removeWhitespaces
        @package: sqlang_structured.py
        @description: 递归地从解析结果中移除空白符号。
        @param tok (sqlparse.sql.TokenList): 解析结果中的一个TokenList对象。
        @return: 无
        @analysis:
            @original: 代码逻辑
            @explanation:
                1. 判断当前的tok是否为TokenList类型，如果是，则执行以下操作：
                   - 创建一个临时列表tmpChildren，用于存储不是空白符的子节点。
                   - 遍历tok中的每个子节点c，如果c不是空白符，则将其添加到tmpChildren中。
                   - 将tok的子节点列表更新为tmpChildren，即移除了空白符子节点。
                   - 对tok的每个子节点c，递归调用removeWhitespaces方法，以移除其下的空白符。
                2. 如果当前tok不是TokenList类型，则不执行任何操作。
        @changes: 无
    '''
    def removeWhitespaces(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            tmpChildren = []
            for c in tok.tokens:
                if not c.is_whitespace:
                    tmpChildren.append(c)

            tok.tokens = tmpChildren
            for c in tok.tokens:
                self.removeWhitespaces(c)
    '''
    @method: identifySubQueries
        @package: sqlang_structured.py
        @description: 递归地识别解析结果中的子查询。
        @param tokenList (sqlparse.sql.TokenList): 解析结果中的一个TokenList对象。
        @return: isSubQuery (bool): 是否存在子查询。
        @analysis:
            @original: 代码逻辑
            @explanation:
                1. 初始化isSubQuery为False。
                2. 遍历tokenList中的每个tok：
                   - 如果tok是TokenList类型，则递归调用identifySubQueries方法，并将返回值赋给subQuery。
                     - 如果subQuery为True且tok是Parenthesis类型，则将tok的ttype设置为SUBQUERY，表示识别到了子查询。
                   - 如果tok的字符串表示为"select"，则将isSubQuery设置为True。
                3. 返回isSubQuery。
        @changes: 无
    '''
    def identifySubQueries(self, tokenList):
        isSubQuery = False

        for tok in tokenList.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                subQuery = self.identifySubQueries(tok)
                if (subQuery and isinstance(tok, sqlparse.sql.Parenthesis)):
                    tok.ttype = SUBQUERY
            elif str(tok) == "select":
                isSubQuery = True
        return isSubQuery

    '''
    @method: identifyLiterals
        @package: sqlang_structured.py
        @description: 识别解析结果中的字面值类型。
        @param tokenList (sqlparse.sql.TokenList): 解析结果中的一个TokenList对象。
        @return: 无
        @analysis:
            @original: 代码逻辑
            @explanation:
                1. 定义blankTokens和blankTokenTypes变量，分别存储空白标记的类型。
                2. 遍历tokenList中的每个tok：
                   - 如果tok是TokenList类型，将其ptype设置为INTERNAL，并递归调用identifyLiterals方法。
                   - 如果tok的ttype为Keyword类型或其字符串表示为"select"，将tok的ttype设置为KEYWORD。
                   - 如果tok的ttype为Integer类型或Literal.Number.Integer类型，将tok的ttype设置为INTEGER。
                   - 如果tok的ttype为Hexadecimal类型或Literal.Number.Hexadecimal类型，将tok的ttype设置为HEX。
                   - 如果tok的ttype为Float类型或Literal.Number.Float类型，将tok的ttype设置为FLOAT。
                   - 如果tok的ttype为String.Symbol类型、String.Single类型或Literal.String.Single类型或Literal.String.Symbol类型，将tok的ttype设置为STRING。
                   - 如果tok的ttype为Wildcard类型，将tok的ttype设置为WILDCARD。
                   - 如果tok的ttype在blankTokens中或tok的类型是blankTokenTypes中的类型，将tok的ttype设置为COLUMN。
        @changes: 无
    '''

    '''
    @method: identifyLiterals
    @package: sqlang_structured.py
    @description: 识别解析结果中的字面类型。
    @param tokenList (sqlparse.sql.TokenList): 解析结果中的 TokenList 对象。
    @return: None
    @analysis:
        @original: 代码逻辑
        @explanation:
            1. 定义 blankTokens 和 blankTokenTypes 变量以存储空白标记的类型。
            2. 遍历 tokenList 中的每个 tok：
               - 如果 tok 是 TokenList 类型，则将其 ptype 设置为 INTERNAL，并递归调用 identifyLiterals 方法。
               - 如果 tok 的 ttype 是 Keyword 类型或其字符串表示为 "select"，则将 tok 的 ttype 设置为 KEYWORD。
               - 如果 tok 的 ttype 是 Integer 类型或 Literal.Number.Integer 类型，则将 tok 的 ttype 设置为 INTEGER。
               - 如果 tok 的 ttype 是 Hexadecimal 类型或 Literal.Number.Hexadecimal 类型，则将 tok 的 ttype 设置为 HEX。
               - 如果 tok 的 ttype 是 Float 类型或 Literal.Number.Float 类型，则将 tok 的 ttype 设置为 FLOAT。
               - 如果 tok 的 ttype 是 String.Symbol 类型、String.Single 类型、Literal.String.Single 类型或 Literal.String.Symbol 类型，则将 tok 的 ttype 设置为 STRING。
               - 如果 tok 的 ttype 是 Wildcard 类型，则将 tok 的 ttype 设置为 WILDCARD。
               - 如果 tok 的 ttype 在 blankTokens 中或 tok 的类型在 blankTokenTypes 中，则将 tok 的 ttype 设置为 COLUMN。
    @changes: 无

    '''
    def identifyLiterals(self, tokenList):
        blankTokens = [sqlparse.tokens.Name, sqlparse.tokens.Name.Placeholder]
        blankTokenTypes = [sqlparse.sql.Identifier]

        for tok in tokenList.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                tok.ptype = INTERNAL
                self.identifyLiterals(tok)
            elif (tok.ttype == sqlparse.tokens.Keyword or str(tok) == "select"):
                tok.ttype = KEYWORD
            elif (tok.ttype == sqlparse.tokens.Number.Integer or tok.ttype == sqlparse.tokens.Literal.Number.Integer):
                tok.ttype = INTEGER
            elif (tok.ttype == sqlparse.tokens.Number.Hexadecimal or tok.ttype == sqlparse.tokens.Literal.Number.Hexadecimal):
                tok.ttype = HEX
            elif (tok.ttype == sqlparse.tokens.Number.Float or tok.ttype == sqlparse.tokens.Literal.Number.Float):
                tok.ttype = FLOAT
            elif (tok.ttype == sqlparse.tokens.String.Symbol or tok.ttype == sqlparse.tokens.String.Single or tok.ttype == sqlparse.tokens.Literal.String.Single or tok.ttype == sqlparse.tokens.Literal.String.Symbol):
                tok.ttype = STRING
            elif (tok.ttype == sqlparse.tokens.Wildcard):
                tok.ttype = WILDCARD
            elif (tok.ttype in blankTokens or isinstance(tok, blankTokenTypes[0])):
                tok.ttype = COLUMN


    '''
    @method: identifyFunctions
        @package: sqlang_structured.py
        @description: 识别解析结果中的函数类型。
        @param tokenList (sqlparse.sql.TokenList): 解析结果中的 TokenList 对象。
        @return: None
        @analysis:
            @original: 代码逻辑
            @explanation:
                1. 遍历 tokenList 中的每个 tok：
                   - 如果 tok 是 Function 类型，则将 parseTreeSentinel 设置为 True。
                   - 如果 tok 是 Parenthesis 类型，则将 parseTreeSentinel 设置为 False。
                   - 如果 parseTreeSentinel 为 True，则将 tok 的 ttype 设置为 FUNCTION。
                   - 如果 tok 是 TokenList 类型，则递归调用 identifyFunctions 方法。
        @changes: 无
    '''
    def identifyFunctions(self, tokenList):
        for tok in tokenList.tokens:
            if (isinstance(tok, sqlparse.sql.Function)):
                self.parseTreeSentinel = True
            elif (isinstance(tok, sqlparse.sql.Parenthesis)):
                self.parseTreeSentinel = False
            if self.parseTreeSentinel:
                tok.ttype = FUNCTION
            if isinstance(tok, sqlparse.sql.TokenList):
                self.identifyFunctions(tok)

    '''
    @method: identifyTables
        @package: sqlang_structured.py
        @description: 识别解析结果中的表格类型。
        @param tokenList (sqlparse.sql.TokenList): 解析结果中的 TokenList 对象。
        @return: None
        @analysis:
            @original: 代码逻辑
            @explanation:
                1. 如果 tokenList 的 ptype 属性为 SUBQUERY，则将 False 添加到 tableStack 中。
                2. 遍历 tokenList.tokens 中的每个 tok：
                   - 如果 tok 是 "." 且 tok 的 ttype 是 Punctuation 且 prevtok 的 ttype 是 COLUMN，则将 prevtok 的 ttype 设置为 TABLE。
                   - 如果 tok 是 "from" 且 tok 的 ttype 是 Keyword，则将 tableStack[-1] 设置为 True。
                   - 如果 tok 是 "where" 或 "on" 或 "group" 或 "order" 或 "union" 且 tok 的 ttype 是 Keyword，则将 tableStack[-1] 设置为 False。
                   - 如果 tok 是 TokenList 类型，则递归调用 identifyTables 方法。
                   - 如果 tok 的 ttype 是 COLUMN 且 tableStack[-1] 为 True，则将 tok 的 ttype 设置为 TABLE。
                3. 如果 tokenList 的 ptype 属性为 SUBQUERY，则从 tableStack 中弹出一个元素。
        @changes: 无
    '''
    def identifyTables(self, tokenList):
        if tokenList.ptype == SUBQUERY:
            self.tableStack.append(False)

        for i in range(len(tokenList.tokens)):
            prevtok = tokenList.tokens[i - 1]
            tok = tokenList.tokens[i]

            if (str(tok) == "." and tok.ttype == sqlparse.tokens.Punctuation and prevtok.ttype == COLUMN):
                prevtok.ttype = TABLE

            elif (str(tok) == "from" and tok.ttype == sqlparse.tokens.Keyword):
                self.tableStack[-1] = True

            elif ((str(tok) == "where" or str(tok) == "on" or str(tok) == "group" or str(tok) == "order" or str(tok) == "union") and tok.ttype == sqlparse.tokens.Keyword):
                self.tableStack[-1] = False

            if isinstance(tok, sqlparse.sql.TokenList):
                self.identifyTables(tok)

            elif (tok.ttype == COLUMN):
                if self.tableStack[-1]:
                    tok.ttype = TABLE

        if tokenList.ptype == SUBQUERY:
            self.tableStack.pop()

    def __str__(self):
        return ' '.join([str(tok) for tok in self.tokens])

    def parseSql(self):
        return [str(tok) for tok in self.tokens]
#############################################################################

#############################################################################
#缩略词处理
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

#获取词性
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

#---------------------子函数1：句子的去冗--------------------
def process_nl_line(line):
    # 句子预处理
    line = revert_abbrev(line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = line.replace('\n', ' ')
    line = line.replace('\t', ' ')
    line = re.sub(' +', ' ', line)
    line = line.strip()
    # 骆驼命名转下划线
    line = inflection.underscore(line)

    # 去除括号里内容
    space = re.compile(r"\([^\(|^\)]+\)")  # 后缀匹配
    line = re.sub(space, '', line)
    # 去除末尾.和空格
    line=line.strip()
    return line


#---------------------子函数1：句子的分词--------------------
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
    #词性标注
    word_tags = pos_tag(cut_words)
    tags_dict = dict(word_tags)
    word_list=[]
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
def sqlang_code_parse(line):
    line = filter_part_invachar(line)
    line = re.sub('\.+', '.', line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = re.sub(' +', ' ', line)

    line = re.sub('>>+', '', line)#新增加
    line = re.sub(r"\d+(\.\d+)+",'number',line)#新增加 替换小数

    line = line.strip('\n').strip()
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    try:
        query = SqlangParser(line, regex=True)
        typedCode = query.parseSql()
        typedCode = typedCode[:-1]
        # 骆驼命名转下划线
        typedCode = inflection.underscore(' '.join(typedCode)).split(' ')

        cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typedCode]
        # 全部小写化
        token_list = [x.lower()  for x in cut_tokens]
        # 列表里包含 '' 和' '
        token_list = [x.strip() for x in token_list if x.strip() != '']
        # 返回列表
        return token_list
    # 存在为空的情况，词向量要进行判断
    except:
        return '-1000'
########################主函数：代码的tokens#################################


#######################主函数：句子的tokens##################################

def sqlang_query_parse(line):
    line = filter_all_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    # 分完词后,再去掉 括号
    for i in range(0, len(word_list)):
        if re.findall('[\(\)]', word_list[i]):
            word_list[i] = ''
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空

    return word_list


def sqlang_context_parse(line):
    line = filter_part_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空
    return word_list

#######################主函数：句子的tokens##################################


if __name__ == '__main__':
    print(sqlang_code_parse('""geometry": {"type": "Polygon" , 111.676,"coordinates": [[[6.69245274714546, 51.1326962505233], [6.69242714158622, 51.1326908883821], [6.69242919794447, 51.1326955158344], [6.69244041615532, 51.1326998744549], [6.69244125953742, 51.1327001609189], [6.69245274714546, 51.1326962505233]]]} How to 123 create a (SQL  Server function) to "join" multiple rows from a subquery into a single delimited field?'))
    print(sqlang_query_parse("change row_height and column_width in libreoffice calc use python tagint"))
    print(sqlang_query_parse('MySQL Administrator Backups: "Compatibility Mode", What Exactly is this doing?'))
    print(sqlang_code_parse('>UPDATE Table1 \n SET Table1.col1 = Table2.col1 \n Table1.col2 = Table2.col2 FROM \n Table2 WHERE \n Table1.id =  Table2.id'))
    print(sqlang_code_parse("SELECT\n@supplyFee:= 0\n@demandFee := 0\n@charedFee := 0\n"))
    print(sqlang_code_parse('@prev_sn := SerialNumber,\n@prev_toner := Remain_Toner_Black\n'))
    print(sqlang_code_parse(' ;WITH QtyCTE AS (\n  SELECT  [Category] = c.category_name\n          , [RootID] = c.category_id\n          , [ChildID] = c.category_id\n  FROM    Categories c\n  UNION ALL \n  SELECT  cte.Category\n          , cte.RootID\n          , c.category_id\n  FROM    QtyCTE cte\n          INNER JOIN Categories c ON c.father_id = cte.ChildID\n)\nSELECT  cte.RootID\n        , cte.Category\n        , COUNT(s.sales_id)\nFROM    QtyCTE cte\n        INNER JOIN Sales s ON s.category_id = cte.ChildID\nGROUP BY cte.RootID, cte.Category\nORDER BY cte.RootID\n'))
    print(sqlang_code_parse("DECLARE @Table TABLE (ID INT, Code NVARCHAR(50), RequiredID INT);\n\nINSERT INTO @Table (ID, Code, RequiredID)   VALUES\n    (1, 'Physics', NULL),\n    (2, 'Advanced Physics', 1),\n    (3, 'Nuke', 2),\n    (4, 'Health', NULL);    \n\nDECLARE @DefaultSeed TABLE (ID INT, Code NVARCHAR(50), RequiredID INT);\n\nWITH hierarchy \nAS (\n    --anchor\n    SELECT  t.ID , t.Code , t.RequiredID\n    FROM @Table AS t\n    WHERE t.RequiredID IS NULL\n\n    UNION ALL   \n\n    --recursive\n    SELECT  t.ID \n          , t.Code \n          , h.ID        \n    FROM hierarchy AS h\n        JOIN @Table AS t \n            ON t.RequiredID = h.ID\n    )\n\nINSERT INTO @DefaultSeed (ID, Code, RequiredID)\nSELECT  ID \n        , Code \n        , RequiredID\nFROM hierarchy\nOPTION (MAXRECURSION 10)\n\n\nDECLARE @NewSeed TABLE (ID INT IDENTITY(10, 1), Code NVARCHAR(50), RequiredID INT)\n\nDeclare @MapIds Table (aOldID int,aNewID int)\n\n;MERGE INTO @NewSeed AS TargetTable\nUsing @DefaultSeed as Source on 1=0\nWHEN NOT MATCHED then\n Insert (Code,RequiredID)\n Values\n (Source.Code,Source.RequiredID)\nOUTPUT Source.ID ,inserted.ID into @MapIds;\n\n\nUpdate @NewSeed Set RequiredID=aNewID\nfrom @MapIds\nWhere RequiredID=aOldID\n\n\n/*\n--@NewSeed should read like the following...\n[ID]  [Code]           [RequiredID]\n10....Physics..........NULL\n11....Health...........NULL\n12....AdvancedPhysics..10\n13....Nuke.............12\n*/\n\nSELECT *\nFROM @NewSeed\n"))



