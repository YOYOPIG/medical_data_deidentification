from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
ws = WS("./data")
pos = POS("./data")
ner = NER("./data")
import re

raw_data_path = 'SampleData_deid.txt'
type_pos = {
    'PERSON' : '',
    'DATE' : 'time',
    'GPE' : 'location',
    'CARDINAL' : 'med_exam',
    'ORDINAL' : '',
    'QUANTITY' : 'med_exam',
    'ORG' : 'location',
    'PRODUCT' : '',
    'EVENT' : '',
    'FAC' : 'location',
    'LANGUAGE' : '',
    'LOC' : 'location',
    'NORP' : 'location',
    'PERCENT' : ''
}

def loadInputFile(path):
    trainingset = list()  # store trainingset [content,content,...]
    position = list()  # store position [article_id, start_pos, end_pos, entity_text, entity_type, ...]
    mentions = dict()  # store mentions[mention] = Type
    with open(raw_data_path, 'r', encoding='utf8') as f:
        file_text=f.read().encode('utf-8').decode('utf-8-sig')
    datas=file_text.split('\n\n--------------------\n\n')[:-1]
    for data in datas:
        data=data.split('\n')
        content=data[0]
        trainingset.append(content)
        annotations=data[1:]
        for annot in annotations[1:]:
            annot=annot.split('\t') #annot= article_id, start_pos, end_pos, entity_text, entity_type
            position.extend(annot)
            mentions[annot[3]]=annot[4]
    
    return trainingset, position, mentions

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")
def CKIP(data):
    result = []
    for i in range(len(data)):
        article_result = []
        para = data[i]
        sentence_list = cut_sent(para)
        word_sentence_list = ws(sentence_list)
        pos_sentence_list = pos(word_sentence_list)
        entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
        for idx in range(len(entity_sentence_list)):
            temp = []
            for j in entity_sentence_list[idx]:
                j = list(j)
                if(j[2] == 'PERSON'):
                    j[2] = type_pos['PERSON']
                elif(j[2] == 'DATE'):
                    j[2] = type_pos['DATE']
                elif(j[2] == 'GPE'):
                    j[2] = type_pos['GPE']
                elif(j[2] == 'CARDINAL'):
                    j[2] = type_pos['CARDINAL']
                elif(j[2] == 'ordinal'):
                    j[2] = type_pos['ordinal']
                elif(j[2] == 'ORDINAL'):
                    j[2] = type_pos['ORDINAL']
                elif(j[2] == 'ORG'):
                    j[2] = type_pos['ORG']
                elif(j[2] == 'PRODUCT'):
                    j[2] = type_pos['PRODUCT']
                elif(j[2] == 'EVENT'):
                    j[2] = type_pos['EVENT']
                elif(j[2] == 'FAC'):
                    j[2] = type_pos['FAC']
                elif(j[2] == 'LANGUAGE'):
                    j[2] = type_pos['LANGUAGE']
                elif(j[2] == 'LOC'):
                    j[2] = type_pos['LOC']
                elif(j[2] == 'NORP'):
                    j[2] = type_pos['NORP']
                else:
                    j[2] = type_pos['PERCENT']
                temp.append(j)
            temp = tuple(temp)
            entity_sentence_list[idx] = temp
        for item in entity_sentence_list:
            for k in item:
                if(item != ()  and k[2] != ''):
                    article_result.append(k)
        result.append(article_result)
    return result

trainingset, position, mentions = loadInputFile(raw_data_path)
result = CKIP(trainingset)
print(result)

