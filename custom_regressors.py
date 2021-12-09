"""
Class and methods for predicting utilities and generate synthethic data
"""
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import fowlkes_mallows_score
from sklearn.ensemble import RandomForestClassifier
import random
import numpy as np
import pdb


class ClusterRegressor(BaseEstimator, ClassifierMixin):
    """Returns the mean value of each cluster of people"""
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.cluster_mean_utility_map = {}

    def fit(self, X, y):
        # X: first column contains ids of clusters, other cols contain the evidence
        # find the best clustering with a validation set
        kmeans_dict = {}
        X_train, X_val, y_train, y_val = train_test_split(X[:, 1:], X[:, 0], test_size=.2, random_state=42)

        # find the best number of clusters with a validation set
        for k in self.num_clusters:
            # padding is necessary as kmeans requires at least two dimensions
            if X_train.shape[1] < 2:
                train_padding = np.zeros((len(X_train), 1), dtype=np.int32)
                val_padding = np.zeros((len(X_val), 1), dtype=np.int32)
                kmeans = KMeans(n_clusters=k, random_state=0).fit(np.append(train_padding, X_train, axis=1))
                y_pred = kmeans.predict(np.append(val_padding, X_val, axis=1))
            else:
                kmeans = KMeans(n_clusters=k, random_state=0).fit(X_train)
                y_pred = kmeans.predict(X_val)
            fm_score = fowlkes_mallows_score(y_val, y_pred)
            kmeans_dict[kmeans] = fm_score

        # retrieve the best clustering
        self.kmeans = max(kmeans_dict, key=kmeans_dict.get)
        #label_clusters = np.unique(self.kmeans.labels_)

        # retrain k-means on the whole trainig set
        if X[:, 1:].shape[1] < 2:
            train_padding = np.zeros((len(X[:, 1:]), 1), dtype=np.int32)
            self.kmeans = KMeans(n_clusters=self.kmeans.cluster_centers_.shape[0], random_state=0).fit(np.append(train_padding, X[:, 1:], axis=1))
        else:
            self.kmeans = KMeans(n_clusters=self.kmeans.cluster_centers_.shape[0], random_state=0).fit(X[:, 1:])
        label_clusters = np.unique(self.kmeans.labels_)
        for lbl in label_clusters:
            idx_label_clusters = np.where(self.kmeans.labels_==lbl)[0]
            self.cluster_mean_utility_map[lbl] = np.mean(y[idx_label_clusters], axis=0).tolist()
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "kmeans")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        # padding is necessary when X contains only one column of evidence
        if X.shape[1] < 2:
            padding = np.zeros((len(X),1), dtype=np.int32)
            pred_clust_labels = self.kmeans.predict(np.append(padding, X, axis=1))
        else:
            pred_clust_labels = self.kmeans.predict(X)

        output = [self.cluster_mean_utility_map[lbl] for lbl in pred_clust_labels]
        return np.array(output)


class MeanRegressor(BaseEstimator, ClassifierMixin):
    """The classifier just returns the mean of a given training column"""
    def __init__(self):
        self.y_mean = None

    def fit(self, X, y):
        self.y_mean = np.mean(y)
        return self

    def predict(self, X, y=None):
        if self.y_mean is None:
            raise RuntimeError("You must train classifer before predicting data!")
        else:
            return np.array([self.y_mean]*X.shape[0])


class RandomRegressor(BaseEstimator, ClassifierMixin):
    """The classifier extracts a random int between min and max"""
    def __init__(self, min_value=0, max_value=4, random_state=42):
        self.random_state = random_state
        self.min_value = min_value
        self.max_value = max_value

    def fit(self, X, y=None):
        assert (type(self.random_state) == int), "random_state parameter must be integer"
        self.random_state_ = check_random_state(self.random_state)
        return self
    def predict(self, X, y=None):
        try:
            getattr(self, "random_state_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        n_samples = X.shape[0]
        return self.random_state_.randint(self.min_value, self.max_value+1, n_samples)


class CRFRegressor(BaseEstimator, ClassifierMixin):
    """Returns the mean value of each cluster of people"""
    def __init__(self, num_clusters_list):
        self.num_clusters_list = num_clusters_list
        self.cluster_mean_utility_map = {}
        self.n_estimators = 100

    def perform_clustering(self, X):
        kmeans_dict = {}
        X_train, X_val, y_train, y_val = train_test_split(X[:, 1:], X[:, 0], test_size=.2, random_state=42)

        # find the best number of clusters with a validation set
        for num_clusters in self.num_clusters_list:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X_train)
            y_pred = kmeans.predict(X_val)
            fm_score = fowlkes_mallows_score(y_val, y_pred)
            kmeans_dict[kmeans] = fm_score

        # retrieve the best clustering
        self.kmeans = max(kmeans_dict, key=kmeans_dict.get)
        #label_clusters = np.unique(self.kmeans.labels_)
        # retrain
        self.kmeans = KMeans(n_clusters=self.kmeans.cluster_centers_.shape[0], random_state=0).fit(X[:, 1:])
        label_clusters = np.unique(self.kmeans.labels_)
        for lbl in label_clusters:
            idx_label_clusters = np.where(self.kmeans.labels_==lbl)[0]
            self.cluster_mean_utility_map[lbl] = np.mean(X[idx_label_clusters, 1:], axis=0).tolist()
        return self.kmeans.labels_

    def perform_clustering_old(self, X_train, y_train, X_test, y_test):
        # allena kmeans su vari cluster
        kmeans_dict = {}
        for num_clusters in self.num_clusters_list:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X_train)
            # testa kmeans
            y_pred = kmeans.predict(X_test)
            fm_score = fowlkes_mallows_score(y_test, y_pred)
            #print(fm_score)
            kmeans_dict[kmeans] = fm_score

        # retrieve the best clustering
        self.kmeans = max(kmeans_dict, key=kmeans_dict.get)
        self.kmeans_score = kmeans_dict[self.kmeans]
        label_clusters = np.unique(self.kmeans.labels_)
        for lbl in label_clusters:
            idx_label_clusters = np.where(self.kmeans.labels_==lbl)[0]
            self.cluster_mean_utility_map[lbl] = np.mean(X_train[idx_label_clusters], axis=0).tolist()
        return self.kmeans.labels_


    def fit(self, X, y):
        try:
            getattr(self, "kmeans")
        except AttributeError:
            raise RuntimeError("You must perform the clustering before training the tree classifier!")

        max_depth = X[1]
        # allena rf con max_depth = i
        self.RF = RandomForestClassifier(n_estimators=self.n_estimators, min_samples_split=2, max_depth=max_depth, max_features='auto', random_state=0)
        self.RF.fit(X[0], y)
        return self


    def get_features(self, data_row, prediction_row, estimator_id):
        # data_row is the test set usefuls for taking the evidence and combine it with the predcited in prediction_row
        # see https://stackoverflow.com/questions/49991677/using-randomforestclassifier-decision-path-how-do-i-tell-which-samples-the-clas
        indicators, index_by_tree = self.RF.decision_path(data_row)
        estimator_id = estimator_id
        begin = index_by_tree[estimator_id]
        end = index_by_tree[estimator_id + 1]
        tree_tmp = self.RF.estimators_[estimator_id]
        node_indices_list = [indicators[idx, begin:end].indices for idx in range(len(data_row))]

        feature_list = []
        # for each decision path in node_indices_list
        for row_idx, node_indices in enumerate(node_indices_list):
            features = set()
            feat_idxs = np.where(tree_tmp.tree_.feature[node_indices] >= 0)[0]
            features.update(tree_tmp.tree_.feature[node_indices][feat_idxs].tolist())
            feature_row = list(features)
            feature_list.append(feature_row)
            prediction_row[row_idx, feature_row] = data_row[row_idx, feature_row]
        return feature_list, prediction_row

    """
    def fit(self, X, y):
        # find the best clustering with a validation set
        X_train, X_test, y_train, y_test = train_test_split(X[0], y, test_size=0.33, random_state=42)

        # allena kmeans su vari cluster
        kmeans_dict = {}
        for num_clusters in self.num_clusters_list:
          kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X_train)

          # testa kmeans
          y_pred = kmeans.predict(X_test)
          fm_score = fowlkes_mallows_score(y_test, y_pred)
          #print(fm_score)
          kmeans_dict[kmeans] = fm_score

        # retrieve the best clustering
        self.kmeans = max(kmeans_dict, key=kmeans_dict.get)
        self.kmeans_score = kmeans_dict[self.kmeans]
        label_clusters = np.unique(self.kmeans.labels_)
        for lbl in label_clusters:
            idx_label_clusters = np.where(self.kmeans.labels_==lbl)[0]
            self.cluster_mean_utility_map[lbl] = np.mean(X_train[idx_label_clusters], axis=0).tolist()

        max_depth = X[1]
        # allena rf con max_depth = i
        self.RF = RandomForestClassifier(n_estimators=self.n_estimators, min_samples_split=2, max_depth=max_depth, max_features='log2', random_state=0)
        self.RF.fit(X[0], y)
        return self
    """

    def predict(self, X, y=None):
        try:
            getattr(self, "RF")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        estimator_id = random.choice(range(self.n_estimators))
        # per ogni elemento del test set predicilo
        y_pred = self.RF.predict(X)
        #pdb.set_trace()
        #print(f"{np.unique(y_pred)} - {self.cluster_mean_utility_map.keys()} - {np.unique(self.kmeans.labels_)}")
        mean_utilities_cluster = [self.cluster_mean_utility_map[lbl] for lbl in y_pred]
        features, pred_with_evidence = self.get_features(X, np.array(mean_utilities_cluster), estimator_id)
        return features, pred_with_evidence, np.array(mean_utilities_cluster)


def MLP_Model(input_dim, output_dim, activation='relu', neurons=100, optimizer='Adam'):
  def mlp_regr():
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dropout(0.3))
    #model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(output_dim))
    model.compile(loss='mse', optimizer= optimizer)#, metrics=['accuracy'])
    return model
  return mlp_regr
