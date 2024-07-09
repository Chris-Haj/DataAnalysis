import re
from collections import defaultdict


def read_file(file_path):
    with open(file_path, 'r', encoding=('utf-8')) as file:
        return file.read()


# Q1 find amount of 'I's
def ICounter(text):
    return len(re.findall(r'\bI\b', text))


# Q2 find amount of words that start with an upper case letter
def count_capital_words(text):
    return len(re.findall(r'\b[A-Z][a-z]*\b', text))


# Q3 find all words that contain a hyphen
def findHyphenatedWords(text):
    return re.findall(r'\b[A-Za-z]+-[A-Za-z]+\b', text)


# Q3 a) find amount of words that contain a hyphen
def countHyphenatedWords(text):
    return len(findHyphenatedWords(text))


# Q3 b) store all words that contain a hyphen and count each word.
def hyphenatedWordsCounter(text):
    dic = dict()
    for pair in findHyphenatedWords(text):
        dic[pair] = dic.get(pair, 0) + 1
    return dic


# Q4 find amount of numbers
def count_numbers(text, ):
    numberslist = [int(i) for i in re.findall(r'\b\d+\b', text)]
    return len(numberslist), numberslist

#Q5 find words that are repeated in a row
def find_repeated_words(text):
    repeated = re.findall(r'\b(\w+)\s+\1\b', text)
    with open('duplications.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(repeated))
    return len(repeated), repeated

#Q6 find strings that are contained within double quotes or single quotes
def find_strings(text):
    containedWords = re.findall(r'\'(.*?)\'|\"(.*?)\"', text)
    with open('quotations.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join([word[0] for word in containedWords if word[0] != '']))


if __name__ == '__main__':
    text = read_file('Data.txt')
    repeated = find_repeated_words(text)
    print(f'Repeated words: {repeated}')
