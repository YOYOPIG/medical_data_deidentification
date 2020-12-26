import os
import numpy as np

class DataProcessor():
    def __init__(self):
        return
    
    def preprocess_data(self, data_path, word_vecs):
        data_list, data_article_id_list = self.generate_dataset(data_path)
        embed_list = self.Word2Vector(data_list, word_vecs)
        x_data = self.generate_feature_list(embed_list)
        y_data = self.generate_label_list(data_list)
        return x_data, y_data
    
    def generate_dataset(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data=f.readlines()#.encode('utf-8').decode('utf-8-sig')
        data_list, data_list_tmp = list(), list()
        article_id_list=list()
        idx=0
        for row in data:
            data_tuple = tuple()
            if row == '\n':
                article_id_list.append(idx)
                idx+=1
                data_list.append(data_list_tmp)
                data_list_tmp = []
            else:
                row = row.strip('\n').split(' ')
                data_tuple = (row[0], row[1])
                data_list_tmp.append(data_tuple)
        if len(data_list_tmp) != 0:
            data_list.append(data_list_tmp)
        
        return data_list, article_id_list

    def generate_feature_list(self, embed_list):
        feature_list = list()
        for idx_list in range(len(embed_list)):
            feature_list_tmp = list()
            for idx_tuple in range(len(embed_list[idx_list])):
                feature_dict = dict()
                for idx_vec in range(len(embed_list[idx_list][idx_tuple])):
                    feature_dict['dim_' + str(idx_vec+1)] = embed_list[idx_list][idx_tuple][idx_vec]
                feature_list_tmp.append(feature_dict)
            feature_list.append(feature_list_tmp)
        return feature_list

    def generate_label_list(self, data_list):
        label_list = list()
        for idx_list in range(len(data_list)):
            label_list_tmp = list()
            for idx_tuple in range(len(data_list[idx_list])):
                label_list_tmp.append(data_list[idx_list][idx_tuple][1])
            label_list.append(label_list_tmp)
        return label_list
    
    def Word2Vector(self, data_list, embedding_dict):
        embedding_list = list()

        # No Match Word (unknown word) Vector in Embedding
        seed = rand()
        unk_vector=np.random.rand(*(list(embedding_dict.values())[0].shape))

        for idx_list in range(len(data_list)):
            embedding_list_tmp = list()
            for idx_tuple in range(len(data_list[idx_list])):
                key = data_list[idx_list][idx_tuple][0] # token

                if key in embedding_dict:
                    value = embedding_dict[key]
                else:
                    value = unk_vector
                embedding_list_tmp.append(value)
            embedding_list.append(embedding_list_tmp)
        return embedding_list