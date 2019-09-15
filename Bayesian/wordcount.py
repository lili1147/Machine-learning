# -*- coding: utf-8 -*-
# @Author: leedagou
# @Date:   2019-03-15 23:17:40
# @Last Modified by:   leedagou
# @Last Modified time: 2019-09-15 16:46:21

'''
从邮件中解析出单词，统计词频
'''

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

import csv

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from os import listdir
from os.path import isfile, join

import chardet
import codecs
import re

# look_utf = codecs.lookup('utf-8')


# nltk.download('punkt')
# nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
stopwords_2 = ['message', 'received', 'subject', 'org', 'date', 'http', 'com']

ham_path = 'data/easy_ham/'
spam_path = 'data/spam/'


def get_words(post):
    '''
    统计单词出现的次数，用set,不允许有重复的单词
    '''
    all_words = set(wordpunct_tokenize(post.replace('=\\n', '').lower()))
    # 删选掉stopwords
    ListofWords = [word for word in all_words if word not in stopwords and len(word) > 2]  # 判断解析出来的单词不是停止词并且长度大于2
    ListofWords = [word for word in ListofWords if re.search('[a-zA-z]', word)]
    return ListofWords


def count_words(post):
    '''
    '''
    all_words = set(wordpunct_tokenize(post.replace('=\\n', '').lower()))
    # 删选掉stopwords
    ListofWords = [word for word in all_words if word not in stopwords and len(word) > 2]  # 判断解析出来的单词不是停止词并且长度大于2
    ListofWords = [word for word in ListofWords if re.search('[a-zA-z]', word)]
    return ListofWords


def mail_from_file(path, history_dict):
    '''
    统计历史库中的数据
    '''
    file_num = 0
    cmds_count = 0
    mails = [mail for mail in listdir(path) if isfile(join(path, mail))]
    for mail_name in mails:
        # print(mail_name)
        message = ''

        if mail_name == 'cmds':
            cmds_count += 1
            continue

        ways = ['utf-8', 'gbk', 'gb2312', 'ASCII', 'Unincode', 'Windows-1252']
        for encoding in ways:
            try:
                with codecs.open(path + mail_name, 'r', encoding=encoding) as f:
                    content = f.read()
                    message = content
                    file_num += 1
                    # print(message)
                break
            except Exception as e:
                pass
        words_list = get_words(message)
        for word in words_list:
            if word not in history_dict:
                history_dict[word] = 1
            else:
                history_dict[word] += 1

    return history_dict, file_num


def count_mail_words(mail):
    '''
    统计每篇邮件中的词频
    '''
    words_dic = {}
    ways = ['utf-8', 'gbk', 'gb2312', 'ASCII', 'Unincode', 'Windows-1252']
    for encoding in ways:
        try:
            with codecs.open(mail, 'r', encoding=encoding) as f:
                content = f.read()
                message = content
                # print(message)
            break
        except Exception as e:
            pass
    words_list = count_words(message)
    for word in words_list:
        if word not in words_dic:
            words_dic[word] = 1
        else:
            words_dic[word] += 1

    return words_dic


def CountToCsv(path, word_dictionary, file_num):
    '''
    将垃圾邮件和正常邮件的词频写入csv文件中
    目的：构建训练集，统计垃圾邮件和正常邮件单词出现的次数
    '''
    f = open(path + '.csv', 'w+', encoding='utf-8', newline='')
    writer = csv.writer(f)
    writer.writerow(['name', 'frequency', 'rate'])
    for word in word_dictionary:
        writer.writerow([word, word_dictionary[word], word_dictionary[word] / file_num])


if __name__ == '__main__':
    # spam_words_dic = mail_from_file('test/', spam_words)
    # print(spam_words_dic)
    singel_word_dic = count_mail_words('test/00001.7848dde101aa985090474a91ec93fcf0')
    print(singel_word_dic)
    # CountToCsv('spam', word_dictionary, file_num)

    # ways = ['utf-8', 'gbk', 'gb2312', 'ASCII', 'Unincode', 'TIS-620', 'iso-8859-1']
    # for encoding in ways:
    #     try:
    #         with codecs.open('data/spam/00313.fab744bfd5a128fca39b69df9811c08', 'r', encoding=encoding) as f:
    #             data = f.read()
    #             print(data.decode('utf-8','ignore'))
    #             print(encoding)
    #         break
    #     except Exception as e:
    #         print('fail')
    # with open('data/spam/00313.fab744bfd5a128fca39b69df9811c086', 'r', encoding='utf-8') as f:
    #     while True:
    #         line = f.readline()
    #         if len(line) == 0:
    #             break
    #         print(line, end='')

    # with codecs.open('data/spam/00317.22fe43af6f4c707c4f1bdc56af959a8e', encoding='Windows-1252') as f:
    #     f = f.read()
    #     print(f)
