import os
import numpy as np

class FileProcessor():
    def __init__(self):
        return

    def process_file(self, raw_data_path, result_path):
        dataset, position, mentions = self.loadInputFile(raw_data_path)
        self.CRFFormatData(dataset, position, result_path)

    def process_word_vector(self, vector_path):
        dim = 0
        word_vecs= {}
        with open(vector_path,encoding="utf-8") as f: # cna.cbow.cwe_p.tar_g.512d.0.txt',encoding="utf-8") as f:
            for line in f:
                try:
                    tokens = line.strip().split()

                    # there 2 integers in the first line: vocabulary_size, word_vector_dim
                    if len(tokens) == 2:
                        dim = int(tokens[1])
                        continue
                
                    word = tokens[0] 
                    vec = np.array([ float(t) for t in tokens[1:] ])
                    word_vecs[word] = vec
                except:
                    pass
        print('vocabulary_size: ',len(word_vecs),' word_vector_dim: ',vec.shape)
        return word_vecs

    def loadInputFile(self, path):
        trainingset = list()  # store trainingset [content,content,...]
        position = list()  # store position [article_id, start_pos, end_pos, entity_text, entity_type, ...]
        mentions = dict()  # store mentions[mention] = Type
        with open(path, 'r', encoding='utf8') as f:
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

    def CRFFormatData(self, trainingset, position, path):
        if (os.path.isfile(path)):
            os.remove(path)
        outputfile = open(path, 'a', encoding= 'utf-8')

        # output file lines
        count = 0 # annotation counts in each content
        tagged = list()
        for article_id in range(len(trainingset)):
            trainingset_split = list(trainingset[article_id])
            while '' or ' ' in trainingset_split:
                if '' in trainingset_split:
                    trainingset_split.remove('')
                else:
                    trainingset_split.remove(' ')
            start_tmp = 0
            for position_idx in range(0,len(position),5):
                if int(position[position_idx]) == article_id:
                    count += 1
                    if count == 1:
                        start_pos = int(position[position_idx+1])
                        end_pos = int(position[position_idx+2])
                        entity_type=position[position_idx+4]
                        if start_pos == 0:
                            token = list(trainingset[article_id][start_pos:end_pos])
                            whole_token = trainingset[article_id][start_pos:end_pos]
                            for token_idx in range(len(token)):
                                if len(token[token_idx].replace(' ','')) == 0:
                                    continue
                                # BIO states
                                if token_idx == 0:
                                    label = 'B-'+entity_type
                                else:
                                    label = 'I-'+entity_type
                                
                                output_str = token[token_idx] + ' ' + label + '\n'
                                outputfile.write(output_str)

                        else:
                            token = list(trainingset[article_id][0:start_pos])
                            whole_token = trainingset[article_id][0:start_pos]
                            for token_idx in range(len(token)):
                                if len(token[token_idx].replace(' ','')) == 0:
                                    continue
                                
                                output_str = token[token_idx] + ' ' + 'O' + '\n'
                                outputfile.write(output_str)

                            token = list(trainingset[article_id][start_pos:end_pos])
                            whole_token = trainingset[article_id][start_pos:end_pos]
                            for token_idx in range(len(token)):
                                if len(token[token_idx].replace(' ','')) == 0:
                                    continue
                                # BIO states
                                if token[0] == '':
                                    if token_idx == 1:
                                        label = 'B-'+entity_type
                                    else:
                                        label = 'I-'+entity_type
                                else:
                                    if token_idx == 0:
                                        label = 'B-'+entity_type
                                    else:
                                        label = 'I-'+entity_type

                                output_str = token[token_idx] + ' ' + label + '\n'
                                outputfile.write(output_str)

                        start_tmp = end_pos
                    else:
                        start_pos = int(position[position_idx+1])
                        end_pos = int(position[position_idx+2])
                        entity_type=position[position_idx+4]
                        if start_pos<start_tmp:
                            continue
                        else:
                            token = list(trainingset[article_id][start_tmp:start_pos])
                            whole_token = trainingset[article_id][start_tmp:start_pos]
                            for token_idx in range(len(token)):
                                if len(token[token_idx].replace(' ','')) == 0:
                                    continue
                                output_str = token[token_idx] + ' ' + 'O' + '\n'
                                outputfile.write(output_str)

                        token = list(trainingset[article_id][start_pos:end_pos])
                        whole_token = trainingset[article_id][start_pos:end_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ','')) == 0:
                                continue
                            # BIO states
                            if token[0] == '':
                                if token_idx == 1:
                                    label = 'B-'+entity_type
                                else:
                                    label = 'I-'+entity_type
                            else:
                                if token_idx == 0:
                                    label = 'B-'+entity_type
                                else:
                                    label = 'I-'+entity_type
                            
                            output_str = token[token_idx] + ' ' + label + '\n'
                            outputfile.write(output_str)
                        start_tmp = end_pos

            token = list(trainingset[article_id][start_tmp:])
            whole_token = trainingset[article_id][start_tmp:]
            for token_idx in range(len(token)):
                if len(token[token_idx].replace(' ','')) == 0:
                    continue

                
                output_str = token[token_idx] + ' ' + 'O' + '\n'
                outputfile.write(output_str)

            count = 0
        
            output_str = '\n'
            outputfile.write(output_str)
            ID = trainingset[article_id]

            if article_id%10 == 0:
                print('Total complete articles:', article_id)

        # close output file
        outputfile.close()