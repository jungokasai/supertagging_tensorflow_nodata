import numpy as np
from preprocessing import Tokenizer
from preprocessing import pad_sequences 
import os
import sys

np.random.seed(1234)


class Dataset(object):
    

    def __init__(self, opts):
        
        if opts.task=='POS_models':
            data_dir = 'data/pos_data'
            self.kfold = False
        elif opts.jackknife:
            data_dir = 'data/super_data'
            jk_data_dir = 'data/pos_data'
            path_to_k_fold = os.path.join(jk_data_dir, 'train_y.txt')
            path_to_k_fold_test = os.path.join(jk_data_dir, 'test_y.txt')
            self.kfold = True


        else:
            data_dir = 'data/super_data'
            self.kfold = False 
 
        path_to_text = os.path.join(data_dir, 'train_x.txt')
        path_to_text_test = os.path.join(data_dir, 'test_x.txt')
        path_to_POS = os.path.join(data_dir, 'train_y.txt')
        path_to_POS_test = os.path.join(data_dir, 'test_y.txt')


        
        self.MAX_NB_WORDS = 200000000000

        # first, build index mapping words in the embeddings set
        # to their embedding vector
        

        f_train = open(path_to_text)
        f_test = open(path_to_text_test)

        texts = f_train.readlines()
        nb_train_samples = len(texts)
        texts = texts + f_test.readlines()

        f_train.close()
        f_test.close()
 
        print('length', len(texts))
        
        f_train.close()
        # f_test.close()


        # finally, vectorize the text samples into a 2D integer tensor
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        indicator = tokenizer.cap_indicator(texts)
        num_indicator = tokenizer.num_indicator(texts)
        suffix = tokenizer.suffix_extract(texts)
        suffix_tokenizer = Tokenizer()
        suffix_tokenizer.fit_on_texts(suffix, non_split=True)
        
        suffix_sequences = suffix_tokenizer.texts_to_sequences(suffix, non_split=True)
# debugging
#        for i in xrange(len(sequences)):
#            assert len(sequences[i]) == len(suffix_sequences[i])


        word_index = tokenizer.word_index
        self.word_index = word_index
        suffix_index = suffix_tokenizer.word_index
        print('Found %s unique words.' % len(word_index))
        data =  pad_sequences(sequences, opts, True)
        suffix_data = pad_sequences(suffix_sequences, opts)
        cap_indicator = pad_sequences(indicator, opts)
        num_indicator = pad_sequences(num_indicator, opts)
       

       
        f_train = open(path_to_POS)
        f_test = open(path_to_POS_test)
        texts = f_train.readlines() + f_test.readlines()
        f_train.close()
        f_test.close()
        lab_tokenizer = Tokenizer()
        lab_tokenizer.fit_on_texts(texts)
        lab_sequences = lab_tokenizer.texts_to_sequences(texts)
        tag_index = lab_tokenizer.word_index
        self.tag_index = tag_index
        self.tag_size = len(tag_index)
        print('Found %s unique tags.' % len(tag_index))
        labels = pad_sequences(lab_sequences, opts)
        #labels = np.expand_dims(labels, -1)  do not need it for tensorflow
        
        if opts.jackknife:

            f_train = open(path_to_k_fold)
            f_test = open(path_to_k_fold_test)
            texts = f_train.readlines() + f_test.readlines()
            f_train.close()
            f_test.close()
            jk_tokenizer = Tokenizer()
            jk_tokenizer.fit_on_texts(texts)
            jk_sequences = jk_tokenizer.texts_to_sequences(texts)
            jk_index = jk_tokenizer.word_index
            self.jk_index = jk_index
            self.jk_size = len(jk_index)
            print('Found %s unique jackknife tags.' % len(jk_index))
            jk_labels = pad_sequences(jk_sequences, opts)

        indices = np.arange(nb_train_samples)
        np.random.shuffle(indices)


        nb_validation_samples = data.shape[0] - nb_train_samples 


        self.X_train = data[:-nb_validation_samples][indices]
        if opts.jackknife:
            self.jk_labels = jk_labels[indices]
            self.jk_labels_test = jk_labels[-nb_validation_samples:]

        self.train_cap_indicator = cap_indicator[:-nb_validation_samples][indices]
        self.train_num_indicator = num_indicator[:-nb_validation_samples][indices]
        self.suffix_train = suffix_data[:-nb_validation_samples][indices]
        self.y_train = labels[:-nb_validation_samples][indices]
        self.X_test = data[-nb_validation_samples:]
        self.test_cap_indicator = cap_indicator[-nb_validation_samples:]
        self.test_num_indicator = num_indicator[-nb_validation_samples:]
        
        self.suffix_test = suffix_data[-nb_validation_samples:]
        self.y_test = labels[-nb_validation_samples:]

        if opts.jackknife:
            K = 10 
            k_fold_samples = nb_train_samples//K*K

            print('splitting into {} folds'.format(K))

            self.X_train = self.X_train[:k_fold_samples] # get rid of the remaining examples for kfold 
            self.train_cap_indicator = self.train_cap_indicator[:k_fold_samples]
            self.train_num_indicator = self.train_num_indicator[:k_fold_samples]
            self.suffix_train = self.suffix_train[:k_fold_samples]
            self.y_train = self.y_train[:k_fold_samples]

            self.X_train_k_fold = np.split(self.X_train[:k_fold_samples], K)
            self.train_cap_indicator_k_fold = np.split(self.train_cap_indicator[:k_fold_samples], K)
            self.train_num_indicator_k_fold = np.split(self.train_num_indicator[:k_fold_samples], K)
            self.suffix_train_k_fold = np.split(self.suffix_train[:k_fold_samples], K)
            self.y_train_k_fold = np.split(self.jk_labels[:k_fold_samples], K)

            print('end splitting')

        self.nb_suffix = len(suffix_index)
        self.suffix_embedding_mat = np.random.randn(self.nb_suffix + 1, 10)
        self.nb_words = min(self.MAX_NB_WORDS, len(word_index))
        if opts.embedding_name == 'random':
            np.random.seed(opts.seed)
            self.embedding_matrix = np.random.uniform(-2, 2, size=(self.nb_words + 1, opts.embedding_dim))
        elif opts.embedding_name == 'word2vec':
	    if not opts.embedding_dim == 300: # word2vec is of 300 dim
                sys.exit('error in dim') 
            filename = os.path.join(opts.embedding_name, 'GoogleNews-vectors-negative300.bin')

            import gensim
            
            self.embedding_matrix = np.zeros((self.nb_words + 1, opts.embedding_dim))
            self.word2vec_model = gensim.models.word2vec.Word2Vec.load_word2vec_format(filename, binary=True)
            print('Found %s word vectors.' % len(self.word2vec_model.vocab))
            for word, i in word_index.items():
                if i > self.MAX_NB_WORDS and word in self.word2vec_model.vocab:
                    self.embedding_matrix[i] = self.word2vec_model[word] 
        else:
            self.embeddings_index = {}
            print('Indexing word vectors.')
            f = open(opts.embedding_name)
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
            f.close()

            print('Found %s word vectors.' % len(self.embeddings_index))
           
            self.embedding_matrix = np.zeros((self.nb_words + 1, opts.embedding_dim))
            for word, i in word_index.items():
                if i > self.MAX_NB_WORDS:
                    continue
                embedding_vector = self.embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    if not self.embedding_matrix.shape[1] == len(embedding_vector):
                        sys.exit('error in dim') 
                       
                    self.embedding_matrix[i] = embedding_vector
                       

        # load pre-trained word embeddings into an Embedding layer

        self._index_in_epoch = 0
        self._num_examples = self.X_train.shape[0]
        self._num_test_examples = self.X_test.shape[0]
        self._epoch_completed = 0 
        self._index_in_test = 0
        if opts.jackknife:
            self._num_hold_in_examples = self.X_train_k_fold[0].shape[0]*(K-1)
            self._num_hold_out_examples = self.X_train_k_fold[0].shape[0]
            self.k = 0
        self.opts = opts

    def next_batch(self, batch_size):

        start = self._index_in_epoch
        if self._index_in_epoch >= self._num_examples:
		# iterate until the very end do not throw away
            self._index_in_epoch =  0
            self._epoch_completed+=1
            assert batch_size <= self._num_examples
            return False
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        self.X_train_batch = self.X_train[start:end]
        if self.opts.embed_dropout < 1.0:
            #print('dropout embed')
            p = self.opts.embed_dropout
            for i in xrange(self.X_train_batch.shape[0]):
                known = set(self.X_train_batch[i,:]) - set([0])
                known = list(known)
                for j in known:
                    if np.random.binomial(1, p) == 0: # dropping
                        self.X_train_batch[i,:][self.X_train_batch[i,:]==j]=0

        self.train_cap_indicator_batch = self.train_cap_indicator[start:end]
        self.train_num_indicator_batch = self.train_num_indicator[start:end]
        self.suffix_train_batch = self.suffix_train[start:end]
        self.y_train_batch = self.y_train[start:end]
        if self.opts.jackknife:
            self.jackknife_train_batch = self.jackknife_train_mat[start:end]
            
            
        return True

    def set_k(self):
        
        k = self.k
        self.k +=1 # for the next hold out

        self.X_train_k = np.vstack(self.X_train_k_fold[:k] + self.X_train_k_fold[k+1:])
        self.train_cap_indicator_k = np.vstack(self.train_cap_indicator_k_fold[:k] + self.train_cap_indicator_k_fold[k+1:])
        self.train_num_indicator_k = np.vstack(self.train_num_indicator_k_fold[:k] + self.train_num_indicator_k_fold[k+1:])
        self.suffix_train_k = np.vstack(self.suffix_train_k_fold[:k] + self.suffix_train_k_fold[k+1:])
        self.y_train_k = np.vstack(self.y_train_k_fold[:k] + self.y_train_k_fold[k+1:])
        
        
        self.X_test_k = self.X_train_k_fold[k] 
        self.test_cap_indicator_k = self.train_cap_indicator_k_fold[k]
        self.test_num_indicator_k = self.train_num_indicator_k_fold[k]
        self.suffix_test_k = self.suffix_train_k_fold[k]
        self.y_test_k = self.y_train_k_fold[k]

        print('ready to run k-fold')

    def next_batch_k(self, batch_size):

        start = self._index_in_epoch
        if self._index_in_epoch >= self._num_hold_in_examples:
		# iterate until the very end do not throw away
            self._index_in_epoch =  0
            self._epoch_completed+=1
            assert batch_size <= self._num_hold_in_examples
            return False

        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        self.X_train_batch = self.X_train_k[start:end]
        self.train_cap_indicator_batch = self.train_cap_indicator_k[start:end]
        self.train_num_indicator_batch =self.train_num_indicator_k[start:end]
        self.suffix_train_batch = self.suffix_train_k[start:end]
        self.y_train_batch = self.y_train_k[start:end]
        return True
    def next_test_batch(self, batch_size):

        start = self._index_in_test
        if self._index_in_test >= self._num_test_examples:
		# iterate until the very end do not throw away
            self._index_in_test =  0
            assert batch_size <= self._num_test_examples
            return False
        self._index_in_test += batch_size
        end = self._index_in_test
        self.X_test_batch = self.X_test[start:end]
        self.test_cap_indicator_batch = self.test_cap_indicator[start:end]
        self.test_num_indicator_batch = self.test_num_indicator[start:end]
        self.suffix_test_batch = self.suffix_test[start:end]
        self.y_test_batch = self.y_test[start:end]
        if self.opts.jackknife:
            self.jackknife_test_batch = self.jackknife_test_mat[start:end]
        return True
    def next_test_batch_k(self, batch_size):

        start = self._index_in_test
        k = self.k
        if self._index_in_test >= self._num_hold_out_examples:
		# iterate until the very end do not throw away
            self._index_in_test =  0
            assert batch_size <= self._num_hold_out_examples
            return False
        self._index_in_test += batch_size
        end = self._index_in_test
        self.X_test_batch = self.X_test_k[start:end]
        self.test_cap_indicator_batch = self.test_cap_indicator_k[start:end]
        self.test_num_indicator_batch = self.test_num_indicator_k[start:end]
        self.suffix_test_batch = self.suffix_test_k[start:end]
        self.y_test_batch = self.y_test_k[start:end]
        return True

    def change_to_supertagging(self, new_opts):
        
        self.kfold = False # kfold is done 
        


    ### change embedding

        #if new_opts.embedding_name == 'random':
        #    np.random.seed(new_opts.seed)
        #    self.embedding_matrix = np.random.uniform(-2, 2, size=(self.nb_words + 1, new_opts.embedding_dim))
        #elif new_opts.embedding_name == 'word2vec':
	#    if not new_opts.embedding_dim == 300: # word2vec is of 300 dim
        #        sys.exit('error in dim') 
        #    filename = os.path.join(new_opts.embedding_name, 'GoogleNews-vectors-negative300.bin')

        #    import gensim
        #    
        #    self.embedding_matrix = np.zeros((self.nb_words + 1, new_opts.embedding_dim))
        #    self.word2vec_model = gensim.models.word2vec.Word2Vec.load_word2vec_format(filename, binary=True)
        #    print('Found %s word vectors.' % len(self.word2vec_model.vocab))
        #    for word, i in self.word_index.items():
        #        if i > self.MAX_NB_WORDS and word in self.word2vec_model.vocab:
        #            self.embedding_matrix[i] = self.word2vec_model[word] 
        #else:
        #    self.embeddings_index = {}
        #    print('Indexing word vectors.')
        #    f = open(new_opts.embedding_name)
        #    for line in f:
        #        values = line.split()
        #        word = values[0]
        #        coefs = np.asarray(values[1:], dtype='float32')
        #        self.embeddings_index[word] = coefs
        #    f.close()

        #    print('Found %s word vectors.' % len(self.embeddings_index))
        #   
        #    self.embedding_matrix = np.zeros((self.nb_words + 1, new_opts.embedding_dim))
        #    for word, i in self.word_index.items():
        #        if i > self.MAX_NB_WORDS:
        #            continue
        #        embedding_vector = self.embeddings_index.get(word)
        #        if embedding_vector is not None:
        #            # words not found in embedding index will be all-zeros.
        #            if not self.embedding_matrix.shape[1] == len(embedding_vector):
        #                sys.exit('error in dim') 
        #               
        #            self.embedding_matrix[i] = embedding_vector
        if new_opts.jackknife:
            import pickle
            with open('../k_fold_sec/predictions.pkl', 'rb') as fhand:
                self.jackknife_train_mat = pickle.load(fhand)
            with open('../k_fold_sec/predictions_test.pkl', 'rb') as fhand:
                self.jackknife_test_mat = pickle.load(fhand)
            self.jackknife_train_mat = np.reshape(self.jackknife_train_mat, (-1, self.y_train.shape[1]))
            self.jackknife_test_mat = np.reshape(self.jackknife_test_mat, (-1, self.y_train.shape[1]))
        self._epoch_completed = 0 
        self._index_in_epoch = 0
        self._index_in_test = 0
	self.jack_knife = new_opts.jackknife
    def invert_dict(self, index_dict): 
        return {j:i for i,j in index_dict.items()}

    def test_one(self, sent_idx):

        self.X_test_batch = self.X_test[[sent_idx]]
        self.test_cap_indicator_batch = self.test_cap_indicator[[sent_idx]]
        self.test_num_indicator_batch = self.test_num_indicator[[sent_idx]]
        self.suffix_test_batch = self.suffix_test[[sent_idx]]
        self.y_test_batch = self.y_test[[sent_idx]]
        if self.opts.jackknife:
            self.jackknife_test_batch = self.jackknife_test_mat[[sent_idx]]
        self.idx_to_word = self.invert_dict(self.word_index)
        self.idx_to_tag = self.invert_dict(self.tag_index)
    def seq_to_sent(self, seq):
        return [self.idx_to_word[idx] for idx in seq]
        
    def seq_to_tags(self, seq):
        return [self.idx_to_tag[idx] for idx in seq]

