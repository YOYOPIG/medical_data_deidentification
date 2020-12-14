# from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.tasks.labeling import BiLSTM_Model
from tensorflow.python.keras.callbacks import ModelCheckpoint
import pickle
from tensorflow.python.client import device_lib
import kashgari
from kashgari.utils import load_model
kashgari.config.use_cudnn_cell = True

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
# print(train_x)
# print(train_y)

filepath = "./checkpoints/saved-model-{epoch:02d}-{acc:.2f}.hdf5"
checkpoint_callback = ModelCheckpoint(filepath,
                                      monitor = 'acc',
                                      verbose = 1)

model = BiLSTM_Model()
print('Training...')
model.fit(train_x, train_y, epochs=2, callbacks=[checkpoint_callback])
# model.fit(train_x, train_y, valid_x, valid_y, epochs=50)
model.save('./models/full_model')


# model = load_model('./models/full_model')

# model.tf_model.load_weights('./checkpoints/saved-model-01-0.96.hdf5')
pred = model.predict(test_x)
print(pred)
total=0
for item in pred:
    total+=len(item)
print(total)
total=0
for item in test_x:
    total+=len(item)
print(total)
with open("./output/bert.arr", "wb") as fp:   #Pickling
    pickle.dump(pred, fp)