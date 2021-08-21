import os

import numpy as np
import pickle

def process_embedding(input_path, output_path,  vocabulary):
  import gensim
  embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(
      input_path, binary=True)
  embedding_matrix_dict = {}
  for v in vocabulary:
      try:
          embedding_matrix_dict[v] = embeddings_index[v]
      except Exception as e:
          embedding_matrix_dict[v] = [0]*300
  print("embedding_matrix_dict = ", embedding_matrix_dict)
  with open(output_path, 'wb') as f:
      pickle.dump(embedding_matrix_dict, f)
  # os._exit(-1)

def process_glove_embedding(input_path, output_path,  vocabulary):
  embedding_index = {}
  for v in vocabulary:
      embedding_index[v] = [0]*300
  with open(input_path, 'r', encoding='utf-8') as f:
    cnt = 1

    for data in f:
      if cnt % 20000 == 0:
        print(cnt)
      cnt += 1
      arr = data.split(' ')
      word = arr[0]
      if word in vocabulary:
        embed = list(map(float, arr[1:]))
        embedding_index[word] = embed

  print("embedding_index = ", embedding_index)
  with open(output_path, 'wb') as f:
      pickle.dump(embedding_index, f)

def get_vocab(path):
  vocabulary = []
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for l in lines:
      word = l.split(' ')[0]
      vocabulary.append(word)
  print(vocabulary, "\n", len(vocabulary))
  return vocabulary

import utils
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def cal_cos_distance(vec1, vec2):
  dist = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
  return dist


def contruct_graph(train_url, test_url, vocabulary, output_path):
  size = len(vocabulary)
  train_set, train_data_mat, \
  train_count, train_label = utils.data_set4(train_url, size)
  test_set, test_data_mat, \
  test_count, test_label = utils.data_set4(test_url, size)
  data_mat = []  
  data_mat.extend(train_data_mat)
  data_mat.extend(test_data_mat)
  
  n = train_data_mat.shape[0]
  print("n = ", n)
  distance_index = []
  for i, d1 in enumerate(train_data_mat):
    distance = []
    for j, d2 in enumerate(train_data_mat):
      distance.append(cal_cos_distance(d1, d2))
    distance_index.append(np.argsort(distance)[-10:-2])
  distance_index = np.array(distance_index)
  print(distance_index)
  with open(output_path, 'wb') as f:
    pickle.dump(distance_index, f)

def test():
  # pass
  vec1 = [1, 2, 3, 4]
  vec2 = [5, 6, 7, 8]
  print(cal_cos_distance(vec1, vec1))
  os._exit(-1)



if __name__ == '__main__':

  data_dir = "data/20news"
  train_url =  os.path.join(data_dir, "train.feat")
  test_url = os.path.join(data_dir, "test.feat")
  vocabulary_path = os.path.join(data_dir, "vocab.new")
  vocabulary = get_vocab(vocabulary_path)
  output_path = os.path.join(data_dir, "train_neighbors.pickle")

  contruct_graph(train_url, test_url, vocabulary, output_path)

