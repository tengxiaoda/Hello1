# coding:utf-8
import sys

sys.path.append('./textrank4zh')

from textrank4zh.TextRank4Sentence import TextRank4Sentence

# textrank of textrank4zh
tr4s = TextRank4Sentence()


def summarize_4zh(text, num=3):
    """
    相似度计算方法：分子部分的意思是同时出现在两个句子中的相同词的个数，
    分母是对句子中词的个数求对数之和。分母这样设计可以遏制较长的句子在相似度计算上的优势
    """
    tr4s.analyze(text=text, lower=True, source='all_filters')
    key_tr4s = tr4s.get_key_sentences(num=num)
    key_tr4s = sorted(key_tr4s, key=lambda i: i.index)

    result = ''
    for term in key_tr4s:
        result += term.sentence
    return result


def textrank_top(text, num=12, top=2):
    """
    对文章进行textrank，之后，再加上文章的前两句
    """
    tr4s.analyze(text=text, lower=True, source='all_filters')
    key_tr4s = tr4s.get_key_sentences(num=999999)

    results = key_tr4s[:num]

    len_text = 0
    for item in results:
        len_text += len(item['sentence'])
    if len_text < 512:
        for i in range(num, len(key_tr4s)):
            if len_text >= 512:
                break
            results.append(key_tr4s[i])
            len_text += len(key_tr4s[i]['sentence'])

    is_find_top = 0
    for i, item in enumerate(results):
        if item['index'] < top:
            is_find_top += 1
    if is_find_top < top:
        results = results[:-(top - is_find_top)]
        key_tr4s.reverse()
        for item in key_tr4s:
            if item['index'] < top:
                results.append(item)
                is_find_top += 1
            if is_find_top >= top:
                break

    results = sorted(results, key=lambda i: i.index)
    return results


def LCS(s1, s2):
    size1 = len(s1) + 1
    size2 = len(s2) + 1
    # 程序多加一行，一列，方便后面代码编写
    chess = [[["", 0] for j in list(range(size2))] for i in list(range(size1))]
    for i in list(range(1, size1)):
        chess[i][0][0] = s1[i - 1]
    for j in list(range(1, size2)):
        chess[0][j][0] = s2[j - 1]
    # print("初始化数据：")
    # print(chess)
    for i in list(range(1, size1)):
        for j in list(range(1, size2)):
            if s1[i - 1] == s2[j - 1]:
                chess[i][j] = ['↖', chess[i - 1][j - 1][1] + 1]
            elif chess[i][j - 1][1] > chess[i - 1][j][1]:
                chess[i][j] = ['←', chess[i][j - 1][1]]
            else:
                chess[i][j] = ['↑', chess[i - 1][j][1]]
    # print("计算结果：")
    # print(chess)
    i = size1 - 1
    j = size2 - 1
    s3 = []
    while i > 0 and j > 0:
        if chess[i][j][0] == '↖':
            s3.append(chess[i][0][0])
            i -= 1
            j -= 1
        if chess[i][j][0] == '←':
            j -= 1
        if chess[i][j][0] == '↑':
            i -= 1
    s3.reverse()
    # print(s3)
    # print("最长公共子序列：%s" % ''.join(s3))
    return len(s3)


def recursive_lcs(str_a, str_b):
    if len(str_a) == 0 or len(str_b) == 0:
        return 0
    if str_a[0] == str_b[0]:
        return recursive_lcs(str_a[1:], str_b[1:]) + 1
    else:
        return max([recursive_lcs(str_a[1:], str_b), recursive_lcs(str_a, str_b[1:])])


def space_efficient_lcs(str_a, str_b):
    """
    longest common subsequence of str_a and str_b, with O(n) space complexity
    """
    if len(str_a) == 0 or len(str_b) == 0:
        return 0
    dp = [0 for _ in range(len(str_b) + 1)]
    for i in range(1, len(str_a) + 1):
        left_up = 0
        dp[0] = 0
        for j in range(1, len(str_b) + 1):
            left = dp[j - 1]
            up = dp[j]
            if str_a[i - 1] == str_b[j - 1]:
                dp[j] = left_up + 1
            else:
                dp[j] = max([left, up])
            left_up = up
    return dp[len(str_b)]


# text = "四海网讯,近日,有媒体报道称:章子怡真怀孕了!报道还援引知情人士消息称,“章子怡怀孕大概四五个月,预产期是年底前后,现在已经不接工作了。”这到底是怎么回事?消息是真是假?针对此消息,23日晚8时30分," \
#        "华西都市报记者迅速联系上了与章子怡家里关系极好的知情人士,这位人士向华西都市报记者证实说:“子怡这次确实怀孕了。她已经36岁了,也该怀孕了。章子怡怀上汪峰的孩子后,子怡的父母亲十分高兴。子怡的母亲," \
#        "已开始悉心照料女儿了。子怡的预产期大概是今年12月底。”当晚9时,华西都市报记者为了求证章子怡怀孕消息,又电话联系章子怡的亲哥哥章子男,但电话通了," \
#        "一直没有人接听。有关章子怡怀孕的新闻自从2013年9月份章子怡和汪峰恋情以来,就被传N遍了!不过,时间跨入2015年,事情却发生着微妙的变化。2015年3月21日,章子怡担任制片人的电影《从天儿降》开机," \
#        "在开机发布会上几张合影,让网友又燃起了好奇心:“章子怡真的怀孕了吗?”但后据证实,章子怡的“大肚照”只是影片宣传的噱头。过了四个月的7月22日,《太平轮》新一轮宣传,章子怡又被发现状态不佳,不时深呼吸," \
#        "不自觉想捂住肚子,又觉得不妥。然后在8月的一天,章子怡和朋友吃饭,在酒店门口被风行工作室拍到了,疑似有孕在身!今年7月11日,汪峰本来在上海要举行演唱会,后来因为台风“灿鸿”取消了。而消息人士称," \
#        "汪峰原来打算在演唱会上当着章子怡的面宣布重大消息,而且章子怡已经赴上海准备参加演唱会了,怎知遇到台风,只好延期,相信9月26日的演唱会应该还会有惊喜大白天下吧。 "
#
# print(summarize_4zh(text, 3))
# print(textrank_top(text, num=3, top=2))
# print(LCS('ABCDF', 'WUPAOWEPAOWFABjkCD;WRIJG;WRIGJWRF'))
# print(recursive_lcs('ABCDF', 'WUPAOWEPAOWFABjkCD;WRIJG;WRIGJWRF'))
# print(space_efficient_lcs('ABCDF', 'WUPAOWEPAOWFABjkCD;WRIJG;WRIGJWRF'))
