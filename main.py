from file_processor import FileProcessor
from data_processor import DataProcessor
from output_generator import OutputGenerator
from model import Model 

data_path = './data/sample.data'
raw_data_path = './data/train_2.txt' #train_2.txt
raw_test_data_path = './test_data/test2_fmt.txt'
test_data_path = './test_data/test.data'
vector_path = './trained_vector/cna.cbow.cwe_p.tar_g.512d.0.txt'#fasttext.vec'
output_path = 'output.tsv'
model_path = './models/crf.model'

if __name__ == "__main__":
    file_processor = FileProcessor()
    file_processor.process_file(raw_data_path, data_path)
    file_processor.process_file(raw_test_data_path, test_data_path)

    word_vecs = file_processor.process_word_vector(vector_path)
    data_processor = DataProcessor()
    x_train, y_train = data_processor.preprocess_data(data_path, word_vecs)
    x_test, y_test = data_processor.preprocess_data(test_data_path, word_vecs)
    testdata_list, test_data_article_id_list = data_processor.generate_dataset(test_data_path)
    input('...')
    
    print('Training model...')
    model = Model()
    # model.train(x_train, y_train)
    model.load(model_path)
    y_pred = model.predict(x_test)
    print(y_pred)

    output = OutputGenerator()
    output.generate(y_pred, testdata_list, testdata_article_id_list, output_path)
