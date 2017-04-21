
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.validation import check_is_fitted
import heapq
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.decomposition import TruncatedSVD
import dill as pickle
import json


class DistanceEstimator(BaseEstimator, TransformerMixin):
  def __init__(self, classes, metric='cosine'):
    self.metric_ = metric
    self.classes_ = classes
    pass

  def fit(self, X, y):
    self.X_ = X
    self.y = y
    return self

  def predict(self, X):
    """ Assign the page closest to the question
    
    Args:
      X = np.array-like of strings
    """

    # Check is fit had been called
    check_is_fitted(self, ['X_', 'y'])

    return [
      heapq.nsmallest(3,
                      zip(
                        (self.classes_[label] for label in self.y),
                        pairwise_distances(self.X_, x,
                                           metric=self.metric_).tolist()
                      ),
                      key=lambda t: t[1])
      for x in X]


def train_model(path='accre_links.jl'):
  with open(path, 'r') as f:
    df = pd.read_json('[' + ",".join(f.readlines()) + ']')

  # Get rid of blank content pages
  df = df.loc[df.content.apply(lambda s: len(s.strip()) > 10), :].reindex()

  # Strip title
  df['title'] = df.title.apply(lambda s: s.strip())

  df['title_url'] = list(zip(df.title, df.url))

  le = LabelEncoder()
  y = le.fit_transform(df.title_url)


  # ('proj', GaussianRandomProjection()),

  model = Pipeline([
    ('vec', TfidfVectorizer(stop_words='english')),
    ('proj', GaussianRandomProjection()),
    ('clf', DistanceEstimator(classes=le.classes_))
  ])

  model.fit(df.content, y)

  return model

def save_model(model):
  with open('page_recommender.dpkl', 'wb') as pickle_output:
    pickle.dump(model, pickle_output)


def load_model(path):
  with open(path, 'rb') as pickle_input:
    return pickle.load(pickle_input)


def test_model(model=None):

  if model is None:
    model = train_model()

  with open('resources/test_questions.json', 'rb') as f:
    questions = json.load(f)['questions'] 

  for q, pred in zip(questions,
                     model.predict(questions)):
    print(q)
    print(pred)
    print("-"*80)

  pass


def test_load_model():
  p = 'page_recommender.dpkl'
  model = load_model(p)
  test_model(model)


if __name__ == "__main__":
  clf = train_model()
  save_model(clf)
