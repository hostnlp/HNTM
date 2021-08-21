from __future__ import print_function

import heapq
import numpy as np
import tensorflow as tf
import math
import os
import utils as utils
import os,sys
sys.path.append('utils')

flags = tf.app.flags
flags.DEFINE_string('data_dir', 'data/20news', 'Data dir path.')
flags.DEFINE_string('dataset', '20news', 'Dataset name.')
flags.DEFINE_float('learning_rate', 5e-4, 'Learning rate.')
flags.DEFINE_float('annealing_rate', 0.01, 'annealing rate.')
flags.DEFINE_float('discrete_rate', 0.1, 'discret rate.')
flags.DEFINE_float('balance_rate', 0.01, 'balance rate.')
flags.DEFINE_float('manifold_rate', 0.3, 'manifold regularization rate.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('n_epoch', 150, 'Number of epochs.')
flags.DEFINE_integer('n_hidden', 256, 'Size of each hidden layer.')
flags.DEFINE_integer('n_topic3', 90, 'Size of topic layer 3.')
flags.DEFINE_integer('n_topic2', 30, 'Size of topic layer 2.')
flags.DEFINE_integer('n_topic1', 10, 'Size of topic layer 1.')
flags.DEFINE_integer('vocab_size', 2000, 'Vocabulary size.')
flags.DEFINE_boolean('test', True, 'Process test data.')
flags.DEFINE_string('non_linearity', 'sigmoid', 'Non-linearity of the MLP.')
FLAGS = flags.FLAGS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class HNTM(object):

    def __init__(self, 
                 vocab_size,
                 n_hidden,
                 n_topic1,
                 n_topic2,
                 n_topic3,
                 learning_rate,
                 discrete_weight,
                 balance_weight,
                 manifold_weight,
                 batch_size,
                 non_linearity):
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic1 = n_topic1
        self.n_topic2 = n_topic2
        self.n_topic3 = n_topic3
        self.non_linearity = non_linearity
        self.learning_rate = learning_rate
        self.discrete_weight = discrete_weight
        self.balance_weight = balance_weight
        self.manifold_weight = manifold_weight
        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, [None, vocab_size], name='input')
        self.mask = tf.placeholder(tf.float32, [None], name='mask')  # mask paddings
        self.train_neighbors_ind = tf.placeholder(tf.int32, [batch_size, None])
        self.kl_weight = tf.placeholder(tf.float32, name='kl_weight')


        with tf.variable_scope('encoder'):
            self.enc_vec = utils.mlp(self.x, [self.n_hidden], self.non_linearity)
            self.mean = utils.linear(self.enc_vec, self.n_hidden, scope='mean')
            self.logsigm = utils.linear(self.enc_vec,
                                     self.n_hidden, 
                                     bias_start_zero=True,
                                     matrix_start_zero=True,
                                     scope='logsigm')
            self.kld = -0.5 * tf.reduce_sum(1 - tf.square(self.mean) + 2 * self.logsigm - tf.exp(2 * self.logsigm), 1)
            self.kld = (self.mask*self.kld)  # mask paddings
            self.weight = tf.nn.softmax(utils.mlp(self.enc_vec, [self.n_hidden, 3], self.non_linearity, scope='weight'),
                                      axis=1)
        with tf.variable_scope('decoder'):
            eps = tf.random_normal((batch_size, self.n_hidden), 0, 1)
            doc_vec = tf.multiply(tf.exp(self.logsigm), eps) + self.mean
            self.theta3 = tf.nn.softmax(utils.linear(doc_vec, self.n_topic3, scope='theta3'), axis=-1) # (None, K)

            neighbors = tf.nn.embedding_lookup(self.theta3, self.train_neighbors_ind)
            self.doc_vec_threeDim = tf.expand_dims(self.theta3, 1)
            self.d1 = tf.abs(neighbors - self.doc_vec_threeDim)
            self.manifold_loss = tf.reduce_sum(tf.abs(neighbors - self.doc_vec_threeDim), axis=[0, 1, 2])

            topic_vec3 = tf.Variable(tf.glorot_uniform_initializer()((self.n_topic3, self.n_hidden)))
            word_vec = tf.Variable(tf.glorot_uniform_initializer()((self.vocab_size, self.n_hidden)))
            temperature = 1**(1/3)
            beta_mat3 = tf.matmul(topic_vec3,word_vec ,transpose_b=True)/temperature
            self.beta3 =  tf.nn.softmax(beta_mat3, axis=1)
            self.logits3 = tf.matmul(self.theta3, self.beta3)

            dep_vec21 = tf.Variable(tf.glorot_uniform_initializer()((self.n_topic3, self.n_hidden)))
            dep_vec22 = tf.Variable(tf.glorot_uniform_initializer()((self.n_topic2, self.n_hidden)))
            self.depend2 = tf.nn.softmax(tf.matmul(dep_vec21, dep_vec22, transpose_b=True), axis=1)
            self.theta2 = tf.matmul(self.theta3, self.depend2)
            topic_vec2 = tf.Variable(tf.glorot_uniform_initializer()((self.n_topic2, self.n_hidden)))
            beta_mat2 = tf.matmul(topic_vec2,word_vec ,transpose_b=True)
            self.beta2 = tf.nn.softmax(beta_mat2, axis=1)
            self.logits2 = tf.matmul(self.theta2, self.beta2, transpose_b=False)

            dep_vec11 = tf.Variable(tf.glorot_uniform_initializer()((self.n_topic2, self.n_hidden)))
            dep_vec12 = tf.Variable(tf.glorot_uniform_initializer()((self.n_topic1, self.n_hidden)))
            self.depend1 = tf.nn.softmax(tf.matmul(dep_vec11, dep_vec12, transpose_b=True), axis=1)
            self.theta1 = tf.matmul(self.theta2, self.depend1)
            topic_vec1 = tf.Variable(tf.glorot_uniform_initializer()((self.n_topic1, self.n_hidden)))
            beta_mat1 = tf.matmul(topic_vec1,word_vec ,transpose_b=True)
            self.beta1 = tf.nn.softmax(beta_mat1, axis=1)
            self.logits1 = tf.matmul(self.theta1, self.beta1, transpose_b=False)

            self.final_logits = tf.multiply( tf.transpose(self.weight[:,0]) , tf.transpose(self.logits1)) + \
                                tf.multiply( tf.transpose(self.weight[:,1]) , tf.transpose(self.logits2))+ \
                                tf.multiply(tf.transpose(self.weight[:, 2]), tf.transpose(self.logits3))
            self.final_logits = tf.log(tf.transpose(self.final_logits))

        self.recons_loss = -tf.reduce_sum(tf.multiply(self.final_logits, self.x), 1)
        self.discrete_loss = 1*(tf.norm(self.depend1,ord=2) + tf.norm(self.depend2,ord=2))
        self.balance_loss = tf.norm(tf.reduce_sum(self.depend1,0), ord=2) + tf.norm(tf.reduce_sum(self.depend2,0), ord=2)
        self.objective = self.recons_loss + self.kl_weight*self.kld  \
                         - self.discrete_weight*self.discrete_loss + self.balance_weight*self.balance_loss + self.manifold_weight*self.manifold_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        fullvars = tf.trainable_variables()
        full_grads = tf.gradients(self.objective, fullvars)
        self.optim_full = optimizer.apply_gradients(zip(full_grads, fullvars))

def train(sess, model, 
          train_url, 
          test_url,
          train_neighbors_url,
          batch_size,
          training_epochs,
          alternate_epochs=5):
  train_set0, train_mat,train_count0 = utils.data_set(train_url,model.vocab_size )
  test_set, test_mat,test_count = utils.data_set(test_url,model.vocab_size)
  dev_size = int(len(train_set0)/20)
  train_size = len(train_set0)-dev_size
  train_set = train_set0[:train_size]
  dev_set = train_set0[train_size:]
  train_count = train_count0[:train_size]
  dev_count = train_count0[train_size:]

  vocab = []
  with open(FLAGS.data_dir + '/' + FLAGS.dataset + '.vocab', 'r') as file_to_read:
      while True:
          lines = file_to_read.readline()
          if not lines:
              break
          word, num = lines.split()
          vocab.append(word)
  vocab_size = len(vocab)
  dev_batches = utils.create_batches(len(dev_set), batch_size, shuffle=False)
  test_batches = utils.create_batches(len(test_set), batch_size, shuffle=False)
  corpus_bow = np.sum(train_mat, 0)
  corpus_topic = corpus_bow / np.linalg.norm(corpus_bow)
  train_neighbors = utils.get_train_neighbors(train_neighbors_url)

  for epoch in range(1,training_epochs):
    train_batches = utils.create_batches(len(train_set), batch_size, shuffle=True)
    kl_weight = min(1,epoch*FLAGS.annealing_rate)
    optim = model.optim_full
    print_mode = 'updating network'
    for i in range(alternate_epochs):
        loss_sum = 0.0
        ppx_sum = 0.0
        kld_sum = 0.0
        NLL_sum = 0.0
        word_count = 0
        doc_count = 0
        for idx_batch in train_batches:
            data_batch, count_batch, mask = utils.fetch_data(
                train_set, train_count, idx_batch,
                vocab_size)
            data_neighbors_idx = utils.get_batch_neighbors_ind(train_neighbors, idx_batch)
            input_feed = {model.x.name: data_batch, model.mask.name: mask,model.kl_weight:kl_weight, model.train_neighbors_ind: data_neighbors_idx}
            _, (all_loss, recon_loss,kld) = sess.run((optim,
                                       [model.objective,model.recons_loss, model.kld]),
                                      input_feed)
            loss = (recon_loss+kld)*mask
            loss_sum += np.sum(loss)
            kld_sum += np.sum(kld) / np.sum(mask)
            word_count += np.sum(count_batch)
            count_batch = np.add(count_batch, 1e-12)
            NLL_sum += np.sum(np.divide(recon_loss, count_batch))
            ppx_sum += np.sum(np.divide(loss, count_batch))
            doc_count += np.sum(mask)
        print_ppx = np.exp(loss_sum / word_count)
        print_ppx_perdoc = np.exp(ppx_sum / doc_count)
        print_NLL_perdoc = np.exp(NLL_sum/ doc_count)
        print_kld = kld_sum / len(train_batches)

        print('| Epoch train: {:d} |'.format(epoch + 1),
              print_mode, '{:d}'.format(i),
              '| Corpus ppx: {:.5f}'.format(print_ppx),  # perplexity for all docs
              '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),  # perplexity for per doc
              '| Per doc NLL: {:.5f}'.format(print_NLL_perdoc),  # perplexity for per doc
              '| KLD: {:.5}'.format(print_kld))
    loss_sum = 0.0
    kld_sum = 0.0
    ppx_sum = 0.0
    word_count = 0
    doc_count = 0
    for idx_batch in dev_batches:
      data_batch, count_batch, mask = utils.fetch_data(
          dev_set, dev_count, idx_batch, FLAGS.vocab_size)
      data_neighbors_idx = utils.get_batch_neighbors_ind(train_neighbors, idx_batch)
      input_feed = {model.x.name: data_batch, model.mask.name: mask,model.kl_weight:1.0 , model.train_neighbors_ind: data_neighbors_idx}
      loss, kld = sess.run([model.objective, model.kld],
                           input_feed)
      recon_loss = sess.run(model.recons_loss,
                            input_feed)
      loss = recon_loss + kld
      loss_sum += np.sum(loss)
      kld_sum += np.sum(kld) / np.sum(mask)
      word_count += np.sum(count_batch)
      count_batch = np.add(count_batch, 1e-12)
      ppx_sum += np.sum(np.divide(loss, count_batch))
      doc_count += np.sum(mask)
    print_ppx = np.exp(loss_sum / word_count)
    print_ppx_perdoc = np.exp(ppx_sum / doc_count)
    print_kld = kld_sum/len(dev_batches)
    print('| Epoch dev: {:d} |'.format(epoch+1),
           '| Perplexity: {:.9f}'.format(print_ppx),
           '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
           '| KLD: {:.5}'.format(print_kld))
    #-------------------------------
    # test
    if FLAGS.test:
      loss_sum = 0.0
      kld_sum = 0.0
      ppx_sum = 0.0
      word_count = 0
      doc_count = 0
      NLL_sum = 0
      for idx_batch in test_batches:
        data_batch, count_batch, mask = utils.fetch_data(
          test_set, test_count, idx_batch, FLAGS.vocab_size)
        data_neighbors_idx = utils.get_batch_neighbors_ind(train_neighbors, idx_batch)
        input_feed = {model.x.name: data_batch, model.mask.name: mask,model.kl_weight:1.0, model.train_neighbors_ind: data_neighbors_idx}
        loss, kld = sess.run([model.objective, model.kld],
                             input_feed)
        recon_loss = sess.run(model.recons_loss,
                              input_feed)
        loss = recon_loss+kld
        loss_sum += np.sum(loss)
        kld_sum += np.sum(kld)/np.sum(mask)
        word_count += np.sum(count_batch)
        count_batch = np.add(count_batch, 1e-12)
        ppx_sum += np.sum(np.divide(loss, count_batch))
        NLL_sum += np.sum(np.divide(recon_loss, count_batch))
        doc_count += np.sum(mask)
      print_ppx = np.exp(loss_sum / word_count)
      print_ppx_perdoc = np.exp(ppx_sum / doc_count)
      print_NLL_perdoc = np.exp(NLL_sum / doc_count)
      print_kld = kld_sum/len(test_batches)
      print('| Epoch test: {:d} |'.format(epoch+1),
             '| Perplexity: {:.9f}'.format(print_ppx),
             '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
            '| Per doc NLL: {:.5f}'.format(print_NLL_perdoc),
             '| KLD: {:.5}\n'.format(print_kld) )
    if epoch %5 == 4:
        tree = build_tree(sess, model)
        print_tree(tree, vocab)
        beta_list = [[],[],[]]
        get_topics(tree, beta_list)
        beta1 = np.array(beta_list[0])
        beta2 = np.array(beta_list[1])
        beta3 = np.array(beta_list[2])
        beta0 = np.concatenate((beta1, np.concatenate((beta2,beta3),axis = 0)), axis=0)
        print(utils.evaluate_coherence(beta0, train_mat, [5, 10, 15]))
        print(utils.evaluate_TU(beta0, [5, 10, 15]))
        topics_specs3 = utils.compute_topic_specialization(beta3, corpus_topic)
        topics_specs2 = utils.compute_topic_specialization(beta2, corpus_topic)
        topics_specs1 = utils.compute_topic_specialization(beta1, corpus_topic)
        print('level 1 topic specialization: ' + str(topics_specs1))
        print('level 2 topic specialization: ' + str(topics_specs2))
        print('level 3 topic specialization: ' + str(topics_specs3))

        CL_set = []
        OL_set = []
        calculate_CLNPMI(tree, train_mat, CL_set, OL_set)
        print('CLNPMI = ', str(np.mean(CL_set)))
        print('overlap = ', str(np.mean(OL_set)))

class Node(object):
    def __init__(self, beta = None,  depth = 0):
        self.beta = beta
        self.childs = []
        self.depth = depth

def build_tree(sess, model):
    root = Node()
    par = np.argmax(sess.run(model.depend1),1)
    par2 = np.argmax(sess.run(model.depend2),1)
    Beta1 = sess.run(model.beta1)
    Beta2 = sess.run(model.beta2)
    Beta3 = sess.run(model.beta3)

    for i,beta1 in enumerate(Beta1):
        childs = par==i
        level1 = Node(beta=beta1, depth=1)
        cnt = 0
        for j, flag in enumerate(childs):
            if flag == 1:
                level2 = Node(beta=Beta2[j], depth=2)
                childs2 = par2 == j
                if np.sum(childs2)>0:
                    for k, flag2 in enumerate(childs2):
                        if flag2 == 1:
                            level3 = Node(beta=Beta3[k], depth=3)
                            level2.childs.append(level3)
                    level1.childs.append(level2)
                    cnt += 1
        if cnt >0:
            root.childs.append(level1)
    return root

def print_tree(Node,vocab):
    if Node.depth != 0:
        phi = Node.beta.tolist()
        words = map(phi.index, heapq.nlargest(10, phi))
        words10 = []
        s = '   '*Node.depth+'level ' +  str(Node.depth)
        for w in words:
            words10.append(vocab[w])
            s += ' '+vocab[w]
        print(s)
    for child in Node.childs:
        print_tree(child,vocab)

def get_topics(Node, beta_list):
    if Node.depth != 0:
        beta_list[Node.depth-1].append(Node.beta.tolist())
    for child in Node.childs:
        get_topics(child, beta_list)

def calculate_CLNPMI(par,train_mat, CL_set, OL_set):
    for child in par.childs:
        if par.depth>0:
            clnpmi = utils.cal_clnpmi(par.beta, child.beta, train_mat)
            overlap = utils.cal_overlap(par.beta, child.beta)
            CL_set.append(clnpmi)
            OL_set.append(overlap)
        calculate_CLNPMI(child, train_mat, CL_set,OL_set)

def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if FLAGS.non_linearity == 'tanh':
      non_linearity = tf.nn.tanh
    elif FLAGS.non_linearity == 'sigmoid':
      non_linearity = tf.nn.sigmoid
    else:
      non_linearity = tf.nn.relu

    hntm = HNTM(vocab_size=FLAGS.vocab_size,
                n_hidden=FLAGS.n_hidden,
                n_topic1=FLAGS.n_topic1,
                n_topic2=FLAGS.n_topic2,
                n_topic3=FLAGS.n_topic3,
                learning_rate=FLAGS.learning_rate,
                discrete_weight=FLAGS.discrete_rate,
                balance_weight=FLAGS.balance_rate,
                manifold_weight=FLAGS.manifold_rate,
                batch_size=FLAGS.batch_size,
                non_linearity=non_linearity)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init = tf.initialize_all_variables()
    sess.run(init)

    train_url = os.path.join(FLAGS.data_dir, 'train.feat')
    test_url = os.path.join(FLAGS.data_dir, 'test.feat')
    train_neighbors_url = os.path.join(FLAGS.data_dir, "train_neighbors.pickle")

    train(sess, hntm, train_url, test_url, train_neighbors_url, FLAGS.batch_size, FLAGS.n_epoch)

if __name__ == '__main__':
    tf.app.run()
