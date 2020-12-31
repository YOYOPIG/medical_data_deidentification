import pickle
from output_generator import OutputGenerator
from data_processor import DataProcessor
from file_processor import FileProcessor

def data4bert():
    f = open("./test_data/final_last.data", encoding='utf-8')
    x=[]
    y=[]
    tmp_x=[]
    tmp_y=[]

    line = f.readline()
    while line:
        if len(line)<2:
            line = f.readline()
            continue
        tmp_x.append(line[0])
        tmp_y.append(line[2:-1])
        # print(tmp_x)
        if(len(tmp_x)>3):
            if(tmp_x[-3:] == ['醫','師','：'] or tmp_x[-3:] == ['民','眾','：'] or tmp_x[-3:] == ['家','屬','：']):
                x.append(tmp_x[:-3])
                y.append(tmp_y[:-3])
                tmp_x = tmp_x[-3:]
                tmp_y = tmp_y[-3:]
        if(len(tmp_x)>4):
            if(tmp_x[-4:] == ['個','管','師','：']):
                x.append(tmp_x[:-4])
                y.append(tmp_y[:-4])
                tmp_x = tmp_x[-4:]
                tmp_y = tmp_y[-4:]
        line = f.readline()
    x.append(tmp_x)
    y.append(tmp_y)
    # for item in x:
    #     print(item)
    with open("./test_data/bert_x.data", "wb") as fp:
        pickle.dump(x, fp)
    with open("./test_data/bert_y.data", "wb") as fp:
        pickle.dump(y, fp)
    # with open("test.txt", "rb") as fp:   # Unpickling
    #     b = pickle.load(fp)
    f.close()

def handle_bert_output():
    f = open("./test_data/final_fmt.txt", encoding='utf-8')
    line = f.readline()
    article_len=[]
    while line:
        article_len.append(len(line))
        try:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
        except:
            pass
    f.close()
    with open("./output/bert.arr", "rb") as fp:   # Unpickling
        arr = pickle.load(fp)
    res = []
    tmp = []
    ctr = 0
    for item in arr:
        prev = len(tmp)
        tmp.extend(item)
        if len(tmp)>=article_len[ctr]-1: #\n behind
            # print('match')
            # print(prev)
            # print(len(tmp))
            # print(article_len[ctr])
            res.append(tmp)
            tmp=[]
            ctr+=1
    print(len(res))
    # print(res)
    article_id=0
    is_tar=False
    start=0
    end=0
    ner_type=''
    output = []
    f = open("./test_data/final_fmt.txt", encoding='utf-8')
    line = f.readline()
    
    for item in res:
        if article_id>=0:
            for i in range(len(item)):
                if not is_tar:
                    if item[i]=='O':
                        continue
                    else:
                        is_tar=True
                        start=i
                        ner_type = item[i][2:]
                else:
                    if item[i]=='O':
                        is_tar=False
                        end = i-1
                        output.append([article_id, start, end, ner_type, line[start:end]])
                    else:
                        continue
        article_id += 1
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
    print(output)
    f.close()
    return res

def format_data():
    f = open("./test_data/final_last.txt", encoding='utf-8')
    lines = []
    line = f.readline()
    line = f.readline()
    while line:
        lines.append(line)
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
    print(len(lines))
    f.close()
    f = open("test_data/final_fmt_last.txt",  "a", encoding='utf-8')
    for line in lines:
        f.write(line)
        f.write('article_id	start_position	end_position	entity_text	entity_type\n')
        f.write('0	0	1	醫師	time\n')
        f.write('\n')
        f.write('--------------------\n')
        f.write('\n')

    f.close()

def check_words():
    f = open("./test_data/final_fmt.txt", encoding='utf-8')
    line = f.readline()
    total=0
    while line:
        total += len(line)-1
        try:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
        except:
            pass
    f.close()
    with open("./output/bert.arr", "rb") as fp:   # Unpickling
        arr = pickle.load(fp)
    total2=0
    for item in arr:
        total2+=len(item)
    
    with open("./test_data/bert_x.data", "rb") as fp:   # Unpickling
        arr = pickle.load(fp)
    total3=0
    for item in arr:
        total3+=len(item)
    print(total)
    print(total2)
    print(total3)

def gen_output():
    test_data_path = './test_data/final_fmt.data'

# file_processor = FileProcessor()
# file_processor.process_file('test_data/final_fmt_last.txt', 'test_data/final_last.data')
# data4bert()
test_data_path = './test_data/final_last.data'
pred = handle_bert_output()
data_processor = DataProcessor()
testdata_list, test_data_article_id_list = data_processor.generate_dataset(test_data_path)
output = OutputGenerator()
output.generate(pred, testdata_list, test_data_article_id_list, 'output.tsv')