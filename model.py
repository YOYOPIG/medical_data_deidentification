import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.externals import joblib

class Model():
    def __init__(self):
        self.MLmodel = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        self.model_path = './models/crf.model'
        return
    
    def train(self, x_train, y_train):
        self.MLmodel.fit(x_train, y_train)
        joblib.dump(self.MLmodel, self.model_path)

    def load(self, path):
        self.MLmodel = joblib.load(path)
    
    def predict(self, x_test):
        y_pred = self.MLmodel.predict(x_test)
        y_pred_mar = self.MLmodel.predict_marginals(x_test)

        # labels = list(self.MLmodel.classes_)
        # labels.remove('O')
        # f1score = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
        # sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0])) # group B and I results
        # print(flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
        # return y_pred, y_pred_mar, f1score
        return y_pred
    
