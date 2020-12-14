from file_processor import FileProcessor
from data_processor import DataProcessor
from output_generator import OutputGenerator
from model import Model 

data_path = './data/sample_simp.data'
raw_data_path = './data/train_2.txt' #train_2.txt
raw_test_data_path = './test_data/final_fmt_simp.txt'
test_data_path = './test_data/test.data'
vector_path = './trained_vector/Tencent_AILab_ChineseEmbedding.txt'#fasttext.vec'
output_path = 'output.tsv'
model_path = './models/crf.model'

if __name__ == "__main__":
    file_processor = FileProcessor()
    # file_processor.process_file(raw_data_path, data_path)
    file_processor.process_file(raw_test_data_path, test_data_path)

    word_vecs = file_processor.process_word_vector(vector_path)
    data_processor = DataProcessor()
    # x_train, y_train = data_processor.preprocess_data(data_path, word_vecs)
    x_test, y_test = data_processor.preprocess_data(test_data_path, word_vecs)
    testdata_list, test_data_article_id_list = data_processor.generate_dataset(test_data_path)
    
    print('Training model...')
    model = Model()
    # model.train(x_train, y_train)
    model.load(model_path)
    y_pred = model.predict(x_test)
    # total=0
    # for item in y_pred:
    #     total+=len(item)
    # print(len(y_pred)) # 159 = num of article
    # print(total) # 270569 words

    output = OutputGenerator()
    output.generate(y_pred, testdata_list, test_data_article_id_list, output_path)
