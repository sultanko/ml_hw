import sklearn.metrics.pairwise as pw
import numpy as np
from collections import Counter

def knn(x, data, labels, k, weight_f='uniform', metric_f='l2', transform_f='id', scaler=None, **params):
    if transform_f == 'id':
        transform_f = lambda x: x
    if weight_f == 'uniform':
        weight_f = lambda x: x/x
    if type(metric_f) is str:
        if metric_f.startswith('minkowski') and len(metric_f) > len('minkowski'):
            params['p'] = int(metric_f[9:])
            metric_f = 'minkowski'
        metric = metric_f
        metric_f = lambda x, y: pw.pairwise_distances([x, y], metric=metric, **params)[0, 1]
    if scaler is not None:
        x = scaler.transform(x.reshape(1, -1)).ravel()
        data = scaler.transform(data)
#    print(weight_f, metric_f, transform_f, scaler, **params)

    mapped_x = transform_f(x)
    f = lambda t: metric_f(mapped_x, transform_f(t))
    arr = np.apply_along_axis(f, 1, data)
    neighbors = np.argsort(arr)[:k]
    neighbors_scores = weight_f(arr[neighbors])
    neighbors_labels = labels[neighbors]
    counter = Counter()
    for l, w in zip(neighbors_labels, neighbors_scores):
        counter[l] += w
    return max(counter.keys(), key=counter.get)

def batchKnn(data, labels, xs=None, k=3, **params):
    if xs is not None:
        classifier = lambda x: knn(x, data, labels, k, **params)
        return np.apply_along_axis(classifier, 1, xs)
    else:
        n = data.shape[0]
        ret = np.zeros(n, np.int32)
        for i in range(n):
            data_ = np.delete(data, i, axis=0)
            labels_ = np.delete(labels, i)
            ret[i] = knn(data[i], data_, labels_, k, **params)
        return ret

class KNN:
    def fit(self, xs, ys, **params):
        self.xs, self.ys = xs, ys
        self.params = params
        return self
    def predict(self, xs):
        return batchKnn(self.xs, self.ys, xs=xs, **self.params)
