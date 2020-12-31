from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.tasks.labeling import BiLSTM_Model
from tensorflow.python.keras.callbacks import ModelCheckpoint
import pickle
from tensorflow.python.client import device_lib
import kashgari
from kashgari.utils import load_model

# from kashgari.embeddings import BertEmbedding
from kashgari.tasks.labeling import BiGRU_Model
from kashgari.embeddings import bert_embedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model
kashgari.config.use_cudnn_cell = True

def bad_handler(x, y):
    ret = y
    ctr = len(ret)
    while ctr<len(x):
        ret.append('O')
        ctr+=1
    return ret


# train_x, train_y = ChineseDailyNerCorpus.load_data('train')
# test_x, test_y = ChineseDailyNerCorpus.load_data('test')
# valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
with open("./data/bert_x.data", "rb") as fp:   # Unpickling
    train_x = pickle.load(fp)
with open("./data/bert_y.data", "rb") as fp:   # Unpickling
    train_y = pickle.load(fp)
with open("./test_data/bert_x.data", "rb") as fp:   # Unpickling
    test_x = pickle.load(fp)
with open("./test_data/bert_y.data", "rb") as fp:   # Unpickling
    test_y = pickle.load(fp)
print(test_x)
input()
############ Train########################
filepath = "./checkpoints/saved-model-{epoch:02d}-{acc:.2f}.hdf5"
checkpoint_callback = ModelCheckpoint(filepath,
                                      monitor = 'acc',
                                      verbose = 1)

my_embedding = kashgari.embeddings.BERTEmbedding('./trained_vector/chinese_L-12_H-768_A-12', task=kashgari.LABELING)
model = BiLSTM_CRF_Model(my_embedding)

# print(train_x)
# print(train_y)

# print(len(test_x)) #16221


model.fit(train_x, train_y, epochs=1)

# model = BiLSTM_Model()
# model.fit(train_x, train_y, epochs=2, callbacks=[checkpoint_callback])
model.save('./models/full_model')
############ Train########################

# model = load_model('./models/full_model')

# model.tf_model.load_weights('./checkpoints/saved-model-01-0.96.hdf5')
# test_x = test_x[8100:]
pred = model.predict(test_x)
print(len(pred))
print(len(test_x))
for i in range(len(test_x)):
    if len(pred[i]) != len(test_x[i]):
        pred[i] = bad_handler(test_x[i], pred[i])
        # print('at conversation #' + str(i))
        # print(pred[i])
        # print(test_x[i])
        # print(len(pred[i]))
        # print(len(test_x[i]))
        # inp = input()
        # if inp=='pass':
        #     break
# total=0
# for item in pred:
#     total+=len(item)
# print(total)
# total=0
# for item in test_x:
#     total+=len(item)
# print(total)
# for x,y in test_x,pred:
#     for i in range(len(y)):
#         if y[i]!='O':
#             print(x[i])

with open("./output/bert.arr", "wb") as fp:   #Pickling
    pickle.dump(pred, fp)