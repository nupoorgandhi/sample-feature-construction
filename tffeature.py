
#1.) filters the tweets for keywords for each county
#2.) Get STI Rates
#2.5) Creates two matrices for each year- term frequency and term frequency binary
#3.) enter those matrices as param for topic modelling
#4.) test with regressor
#tffeature runner
#import sklearn
#from sklearn import feature_extraction
#from sklearn.feature_extraction.text import TfidfVectorizer
#import nupoorg2
import glob
from time import time
import os, gensim
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import csv
import re
import pickle, random
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, PCA
from utils import print_top_words, eng_stopwords
from sklearn.preprocessing import normalize
from gensim.models import LdaModel

FIXED_SEED = 42
# before training/inference:
np.random.seed(FIXED_SEED)

class TFFeature(object):

    #path2rates is a file containing STI rates
    def __init__(self, sti= 'CHLA', path2rates='/home/amorale4/stiRates/', random_state = 420):
        self.sti = sti
        self.path2stis = path2rates
        self.years = ['2009', '2010', '2011', '2012', '2013' ]
        self.tfMatrix = None
        self.RSEED=random_state
        self.ind2vocab = None

    def gen_labels_binary(self, rate, Y):
        l = np.array(Y).astype(np.float)
        med = np.median(l)
        mean = np.mean(l)
        sd = np.std(l)

        if rate > mean:
            return 1
        else:
            return 0

    def gen_labels(self, rate, Y):
        l =  np.array(Y).astype(np.float)
        med = np.median(l)
        mean = np.mean(l)
        sd = np.std(l)
        
        if rate > mean:
            if rate < (mean + sd):
                return 5
            elif rate < (mean + 2*sd):
                return 3
            else:
                return 1
        else:
            if rate > (mean - sd):
                return 4
            elif rate > (mean - 2*sd):
                return 2
            else:
                return 0


    def get_XY(self, datapaths, sti_files, isClass=False):
        ret_files = []
        sti_rates = []
        for i, datapath in enumerate(datapaths):
            print ("[tffeature]: datapath:", datapath )
            generalCorpus = glob.glob(datapath + "/*.tsv")
            stiRatesFilePath = os.path.join( self.path2stis, sti_files[i])
            rates = self.get_rates(generalCorpus, stiRatesFilePath, self.sti)
            #print("[stiRegressor]: countyNames: ",rates[1])
            #print("[stiRegressor]: stiRates: ", rates[0])
            for yr_idx, year in enumerate(self.years):
                if year in datapath.split('_'):
                    break

            finalCorpus, idxs = self.reconstructCorpusFromCountyNames_indices(rates[1], generalCorpus, self.years[yr_idx])
            #print ("[stiRegressor]: finalCopus",finalCorpus)
            ret_files += finalCorpus
            if isClass:
                
                #get labels
                Y = np.array([float(r) for r in np.array(rates[0])[idxs] ])
                Y_log = np.log(Y)
                sti_rates += [self.gen_labels(float(r),Y_log) for r in Y_log]
            else:
                sti_rates += [float(r) for r in np.array(rates[0])[idxs]]

        if len( ret_files ) != len( sti_rates):
            print ("[TFFeature]:get_XY: WARNING: different number for X and Y")
        # print("STI RATES: ",sti_rates)
        if isClass:
            return ret_files, sti_rates
        else:
            return ret_files, np.array(sti_rates)

    def reconstructCorpusFromCountyNames_indices(self, countyNames, allDocuments,year):
        newCorpus = []
        corp_indices = []
        #shorten documents once so dont have to do it everytime you want to compare a new document
        allDocs = [ os.path.basename(doc).split('.')[0] for doc in allDocuments]
        #remove all but potential ids
        allDocs = [ doc[:doc.rfind('_')] for doc in allDocs ]
        # print ("[tffeature]: allDocs", allDocs)
        for i, county in enumerate(countyNames):
            if county in allDocs:
                newCorpus.append( allDocuments[ allDocs.index(county) ] )
                corp_indices.append(i)

        return newCorpus, corp_indices

    def reconstructCorpusFromCountyNames(self, countyNames, allDocuments,year):
        newCorpus = []
        #shorten documents once so dont have to do it everytime you want to compare a new document
        allDocs = [ os.path.basename(doc).split('.')[0] for doc in allDocuments]
        #remove all but potential ids
        allDocs = [ doc[:doc.rfind('_')] for doc in allDocs ]
        # print ("[tffeature]: allDocs", allDocs)
        for county in countyNames:
            if county in allDocs:
                newCorpus.append( allDocuments[ allDocs.index(county) ] )

            # found = False
            # for doc in allDocuments:
            #     shortened = doc
            #     shortened = shortened.replace("Penn_"+year+"_filterKeyword/","")
            #     shortened = shortened[:len(county)]
            #     print(shortened, "  ", county)
            #     if shortened == county:
            #         newCorpus.append(doc)
            #         print("appended to new corpus")
            #         break
        return newCorpus
                

    # county rates is thefilename where the STI rate information is located
    # ll is a list of county names
    def get_rates(self,ll, countyRates, mySTI):
        countyNames = []
        if mySTI == "HIVP":
            sti_key = 4
        elif mySTI == "HIVN":
            sti_key = 3
        elif mySTI == "CHLA":
            sti_key = 8
        elif mySTI == "GONO":
            sti_key = 6
        else: #default is HIVP
            sti_key = 3

        print("[tffeature]: get_rates: (mySTI, sti_key)", mySTI, sti_key)

        if len (ll) < 1:
            print ('WARNING: counties list is empty...')
        #for entry in ll:
         #   l1.append(os.path.basename(entry))
        #print ('[tffeature]: get_rates: ll', ll)
        if '.txt' in ll[0]:
            l1 = [re.sub('[0-9]','',x.split('.txt')[0])[:-1] for x in ll]
        elif '.tsv' in ll[0]:
            #l1 = [re.sub('[0-9]', '', x.split('.tsv')[0])[:-1] for x in ll]
            l1 = [os.path.basename(w).replace('.tsv', "") for w in ll]
            for i in range(len(l1)):
                m = l1[i].rfind('_')
                entry = l1[i]
                l1[i] = entry[:m]
                # print(l1[i], entry)

        else:
            print (ll[0], 'unrecognized format.')
            l1 = ll

        if 'Tazewell_County_IL' in l1:
            print('can find keys')

        print('[tffeature]: get_rates: (l1[0])', l1[0])
        hivrates = []# a list of tuples (county name, rate)
        i = 1
        previous_keys = []

        for ll in csv.reader( open(countyRates, 'r' )):
            if i > 1:
                county_rate = ll[sti_key].strip()
                if len(county_rate) > 0 and (float(county_rate) > 0 ):
                    #print("[tffeature]: get_rates: county_rate", county_rate)
                    county_name = ll[2]
                    state_name = ll  [1]
                    key = re.sub('\s','_',county_name) + "_" + state_name
                    #print("[tffeature]: get_rates: (key-rate)" +key +" "+(county_rate))
                    if key in l1:
                        #print('abc')
                        if key not in previous_keys:
                            #print('def')
                            previous_keys.append(key)
                            hivrates.append((key,(county_rate)))
                        else:
                            print("duplicate key:", key)
            i = i + 1

        index_dict = {item: index for index, item in enumerate(l1)}
        hivrates.sort(key=lambda t: index_dict[t[0]])

        print("end of get_rates")
        hiv = [x[1] for x in hivrates]
        names = [x[0] for x in hivrates]
        #print(hiv)
        #print(names)
        #print("l1", l1)
        print("hiv len", len(hiv))
        print("names len", len(names))
        print('prev keys', len(previous_keys))
        return hiv, names

    def save_models(self,output, lda_model, vectorizer):
        print ("saving model ...")
        with open (output+'.pickle', 'wb') as f:
            pickle.dump({'lda_model': lda_model, 'vectorizer':vectorizer} , f)

    def load_models(self, pklfile):
        with open (pklfile, 'rb') as f:
            data = pickle.load(f)
        return data['lda_model'], data['vectorizer']

    def gen_feature_vectorizer(self,type, min_docFreq = 10, max_docFreq = 0.70, max_totFeatures = 50000, vocab = None,
                               min_ngrams=1, max_ngrams=1, input_type='filename'):
        if type is 'tf':
            return CountVectorizer(input = input_type,stop_words=eng_stopwords, min_df=min_docFreq,ngram_range=(min_ngrams, max_ngrams),
                                   max_features=max_totFeatures, max_df=max_docFreq, vocabulary=vocab)
        if type is 'tfb':
            return CountVectorizer(input = input_type,stop_words=eng_stopwords, min_df=min_docFreq,
                                   max_features=max_totFeatures, max_df=max_docFreq, binary = True, vocabulary=vocab)
        else:
            return TfidfVectorizer(input = input_type,stop_words=eng_stopwords, min_df=min_docFreq,
                                   max_features=max_totFeatures, max_df=max_docFreq, vocabulary=vocab)


    def gen_svd_model (self, vectorizer, n_covar, corpus, update_mat=False):
        if update_mat or self.tfMatrix == None:  # dont want to re-create the everytime we want to generate an lda_model (e.g. different topic numbers)
            # unless this is a new vectorizer, or a new corpus
            t0 = time()
            self.tfMatrix = vectorizer.fit_transform(corpus)
            # transformer = TfidfTransformer()               # this would be a substitue for TFIDFVectorizor, but already using it...
            # self.tfMatrix = transformer.fit_transform(TermDocMatrix)
            print("[tffeature]: gen_svd_model: transform done in %0.3fs." % (time() - t0))
            print("[tffeature]: gen_svd_model: tf shape:", self.tfMatrix.shape)
            self.tfMatrix = normalize(self.tfMatrix, norm='l1', axis=1)


        #pca_model = PCA(n_components=n_covar, whiten=True, random_state=0)
        svd_model = TruncatedSVD(n_components=n_covar, random_state=0, n_iter=20)
        t0 = time()
        svd_model.fit(self.tfMatrix)
        print("[tffeature]: gen_svd_model: svd done in %0.3fs." % (time() - t0))
        #print("\n[tffeature]: gen_lda_model: Topics in LDA model:")
        #print_top_words(pca_model, vectorizer.get_feature_names(), 20)
        return svd_model


    def gen_pca_model (self, vectorizer, n_covar, corpus, update_mat=False):
        if update_mat or self.tfMatrix == None:  # dont want to re-create the everytime we want to generate an lda_model (e.g. different topic numbers)
            # unless this is a new vectorizer, or a new corpus
            t0 = time()
            self.tfMatrix = vectorizer.fit_transform(corpus)
            self.tfMatrix = self.tfMatrix.todense()
            # transformer = TfidfTransformer()               # this would be a substitue for TFIDFVectorizor, but already using it...
            # self.tfMatrix = transformer.fit_transform(TermDocMatrix)
            print("[tffeature]: gen_svd_model: transform done in %0.3fs." % (time() - t0))
            print("[tffeature]: gen_svd_model: tf shape:", self.tfMatrix.shape)
            self.tfMatrix = normalize(self.tfMatrix, norm='l1', axis=1)


        pca_model = PCA(n_components=n_covar, whiten=True, random_state=0)
        #svd_model = TruncatedSVD(n_components=n_covar, random_state=0, n_iter=20)
        t0 = time()
        print('[tffeature]: gen_pca_model: type', type(self.tfMatrix))
        pca_model.fit(self.tfMatrix)
        print("[tffeature]: gen_pca_model: pca done in %0.3fs." % (time() - t0))
        #print("\n[tffeature]: gen_lda_model: Topics in LDA model:")
        #print_top_words(pca_model, vectorizer.get_feature_names(), 20)
        return pca_model

    def gen_lda_model(self, vectorizer, n_topics, finalCorpus, update_mat = False):
        #vectorizer = gen_feature_vectorizer(self,t)
        t0 = time()
        #print(finalCorpus)
        if update_mat or self.tfMatrix == None: # dont want to re-create the everytime we want to generate an lda_model (e.g. different topic numbers)
                                                # unless this is a new vectorizer, or a new corpus
            t0 = time()
            #print('[gen_lda_model] finalCorpus:', len(finalCorpus), type(finalCorpus), len(finalCorpus[0]), finalCorpus.columns, "||", finalCorpus[0][1])
            #try this
            #finalCorpus = ['This is the first document.', 'This is the second second document.', 'And the third one.','Is this the first document?']
            #finalCorpus = finalCorpus.iloc[0:5]
            self.tfMatrix = vectorizer.fit_transform(finalCorpus)
            #transformer = TfidfTransformer()               # this would be a substitue for TFIDFVectorizor, but already using it...
            #self.tfMatrix = transformer.fit_transform(TermDocMatrix)
            print("[tffeature]: gen_lda_model: transform done in %0.3fs." % (time() - t0))
            print("[tffeature]: gen_lda_model: tfMatrix shape:", self.tfMatrix.shape)
            self.tfMatrix = normalize(self.tfMatrix, norm='l1', axis=1)
  

        # rule of thumb:
        # alpha >> beta - we get simpler topic distributions with more topic weight appearing in documents
        # alpha << beta - we get complicated topic distributions with fewer topic weight appearing in documents
        # alpha, beta -  small : we get contension in simpler topics, but also fewer topics in documents
        # alpha, beta - big: contension in complicated topics, but many topics in documents
        # default alpha = beta = 1/n_topics (which depending on number of topics can be big or small)
        alpha = 1.0/float(n_topics)
        beta = (1.0/float(n_topics))**2
        #beta = 1.0/float(n_topics)
       
        lda_model = LatentDirichletAllocation(n_topics=n_topics, max_iter=20,learning_method='online',random_state=0, n_jobs=-1)
        t0 = time()
        lda_model.fit(self.tfMatrix)
        print("[tffeature]: gen_lda_model: topics done in %0.3fs." % (time() - t0))
        print("\n[tffeature]: gen_lda_model: Topics in LDA model:")
        print_top_words(lda_model, vectorizer.get_feature_names(), 20)
        
        return lda_model

    def gen_glda_model(self, vectorizer, n_topics, finalCorpus, update_mat = False, alpha='auto', eta="auto", ittrs=300 ):
        # vectorizer = gen_feature_vectorizer(self,t)
        # print(finalCorpus)
        if update_mat or self.tfMatrix == None:  # dont want to re-create the everytime we want to generate an lda_model (e.g. different topic numbers)
            # unless this is a new vectorizer, or a new corpus
            t0 = time()
            print("finalCorpus type:", type(finalCorpus))
            #print("finalCorpus: ", finalCorpus)
            self.tfMatrix = vectorizer.fit_transform(finalCorpus)
            # transformer = TfidfTransformer()               # this would be a substitue for TFIDFVectorizor, but already using it...
            # self.tfMatrix = transformer.fit_transform(TermDocMatrix)
            print("[tffeature]: gen_lda_model: transform done in %0.3fs." % (time() - t0))
            print("[tffeature]: gen_lda_model: tfMatrix shape:", self.tfMatrix.shape)
            print("[tffeature]: gen_lda_model: vectorizer:", vectorizer)

            self.tfMatrix = normalize(self.tfMatrix, norm='l1', axis=1)
            print('[tffeature]: type after normalize: ',type(self.tfMatrix))

        # row_sums = scipy_sparse_matrix.sum(axis=1)
        # scipy_sparse_matrix = scipy_sparse_matrix / row_sums[:, np.newaxis]
        corpus = gensim.matutils.Sparse2Corpus(self.tfMatrix, documents_columns=False)
        # invert vocabulary
        # idx_to_term
        if self.ind2vocab == None:
            inv_vocabulary = {}
            for w in sorted(vectorizer.vocabulary_):
                inv_vocabulary[vectorizer.vocabulary_[w]] = w
                # if vectorizer.vocabulary_[w] == 0:
                #     print(w)
        self.ind2vocab = inv_vocabulary

        # print(corpus)
        # print(max(inv_vocabulary.keys()))
        # print('scipy shape', self.tfMatrix.shape)
        # print('vocabs: ', len(vectorizer.vocabulary_), len(inv_vocabulary))
        np.random.seed(self.RSEED)
        lda = LdaModel(corpus, num_topics=n_topics, id2word=self.ind2vocab, alpha=alpha, eta=eta,
                       random_state= np.random.RandomState(self.RSEED), iterations=ittrs)
        return lda

    #sorts vocabulary indices
    # eta = 1/num_topics
    def gen_glda_model_sv(self, vectorizer, n_topics, finalCorpus, update_mat = False, alpha='auto', eta="auto", ittrs=300 ):
        # vectorizer = gen_feature_vectorizer(self,t)
        # print(finalCorpus)
        if update_mat or self.tfMatrix == None:  # dont want to re-create the everytime we want to generate an lda_model (e.g. different topic numbers)
            # unless this is a new vectorizer, or a new corpus
            t0 = time()
            self.tfMatrix = vectorizer.fit_transform(finalCorpus)
            # transformer = TfidfTransformer()               # this would be a substitue for TFIDFVectorizor, but already using it...
            # self.tfMatrix = transformer.fit_transform(TermDocMatrix)
            print("[tffeature]: gen_lda_model: transform done in %0.3fs." % (time() - t0))
            print("[tffeature]: gen_lda_model: tfMatrix shape:", self.tfMatrix.shape)

            self.tfMatrix = normalize(self.tfMatrix, norm='l1', axis=1)
            print('[tffeature]: type after normalize: ',type(self.tfMatrix))

        # invert vocabulary
        # idx_to_term
        vocab_key = []
        if self.ind2vocab == None:
            inv_vocabulary = {}
            i = 0
            for w in sorted(vectorizer.vocabulary_):
                #inv_vocabulary[vectorizer.vocabulary_[w]] = w
                inv_vocabulary[i] = w
                i+=1
                vocab_key.append(vectorizer.vocabulary_[w])
                # if vectorizer.vocabulary_[w] == 0:
                #     print(w)
        self.ind2vocab = inv_vocabulary
        self.tfMatrix = self.tfMatrix[:,vocab_key]
        # print(self.tfMatrix[0].toarray().tolist() )
        # print(self.tfMatrix[1].toarray().tolist() )
        # print(self.tfMatrix[2].toarray().tolist() )
        print( self.ind2vocab [0], self.ind2vocab [1], self.ind2vocab [2])
        # row_sums = scipy_sparse_matrix.sum(axis=1)
        # scipy_sparse_matrix = scipy_sparse_matrix / row_sums[:, np.newaxis]
        corpus = gensim.matutils.Sparse2Corpus(self.tfMatrix, documents_columns=False)

        # print(corpus)
        # print(max(inv_vocabulary.keys()))
        # print('scipy shape', self.tfMatrix.shape)
        # print('vocabs: ', len(vectorizer.vocabulary_), len(inv_vocabulary))
        np.random.seed(self.RSEED)
        random.seed(self.RSEED)
        lda = LdaModel(corpus, num_topics=n_topics, id2word=self.ind2vocab, alpha=alpha, eta=1.0/n_topics,
                       random_state= np.random.RandomState(self.RSEED), iterations=ittrs,
                       minimum_probability=0.001, minimum_phi_value=0.001 )
        print('[tffeature]: lda perplexity:', lda.log_perplexity(corpus))
        return lda, vocab_key

    def get_top_N_topic_terms(self, N, lda_model, vectorizer):
        top_words_list = print_top_words(lda_model, vectorizer.get_feature_names(), N)
        flat_top_words_list = set([item for sublist in top_words_list for item in sublist])
        return list(flat_top_words_list)

    def gen_topN_features (self, N, lda_model, vectorizer, X_files):
        tc_vocab = self.get_top_N_topic_terms(N, lda_model, vectorizer)
        tc_vectorizer = self.gen_feature_vectorizer('tf', vocab=tc_vocab, min_docFreq = 1, max_docFreq = 0.99)
        return tc_vectorizer.transform(X_files)

    def gen_topN_gfeatures(self, N, lda_model, vectorizer, X_files):
        #tc_vocab = self.get_top_N_topic_terms(N, lda_model, vectorizer)
        #print('lda', lda_model)
        tc_vocab = []
        for i, topic in enumerate(lda_model.show_topics(num_topics=-1, num_words=20, formatted=False)):
            (topic_number, word_list) = topic
            words = [a for (a, b) in word_list]
            #print('topic #', i, ":", ",".join(words))
            tc_vocab += words
        tc_vocab = list(set(tc_vocab))
        tc_vectorizer = self.gen_feature_vectorizer('tf', vocab=tc_vocab, min_docFreq=1, max_docFreq=0.99)
        return tc_vectorizer.transform(X_files)
        # self.tfMatrix = self.tfMatrix.todense()
        # # transformer = TfidfTransformer()               # this would be a substitue for TFIDFVectorizor, but already using it...
        # # self.tfMatrix = transformer.fit_transform(TermDocMatrix)
        # print("[tffeature]: gen_svd_model: transform done in %0.3fs." % (time() - t0))
        # print("[tffeature]: gen_svd_model: tf shape:", self.tfMatrix.shape)




        #print_top_words(lda_model, tf_vectorizer.get_feature_names(), n_top_words)
        #return tfMatrix,lda_model,theta_X

def gen_model_features(model_type='lda', dt_type = 'tfidf'):
    f = TFFeature()

    # location of csv files containing tweets
    trdp = ['/home/amorale4/tf-stis/transformedData/Penn_2009', '/home/amorale4/tf-stis/transformedData/Penn_2010']
    # tr_sti_files = ['sti2009_v201607.csv', 'sti2010_v201607.csv']
    # X_train_files, Y_train = f.get_XY(trdp, tr_sti_files) # do this when we want to infer topics, since we only care about those which have the sti rates
    # here we dont care about those which do or do not have the sti-rates, instead we are interested in generating topics for counties
    X_train_files = []
    for dataset in trdp:
        X_train_files += glob.glob(dataset + "/*.tsv")

    print ("[tffeature]: number of files", len(X_train_files))

    # we have to create 3 matrices
    #e have to create 3 term-doc matrices (9 in total). Each term-doc matrix has different weighting scheme: binary, term frequency, tfidf.
    k = [5, 10, 15, 20, 30, 40, 50, 100, 150, 200, 250, 300]
    #k = [150, 200]
    vectorizer_tf = f.gen_feature_vectorizer("tf")
    vectorizer_tfb = f.gen_feature_vectorizer("tfb")
    vectorizer_tfidf = f.gen_feature_vectorizer("tfidf")

    print("[tffeature]: BEGINNING")

    if model_type == 'lda':
        output_prefix = '/home/amorale4/tf-stis/models/min10_max70_Topic'
        gen_model = f.gen_lda_model
    elif  model_type == 'glda':
        output_prefix = '/home/amorale4/tf-stis/models/min10_max70_gTopic'
        gen_model = f.gen_glda_model
    elif model_type == 'pca':
        output_prefix = '/home/amorale4/tf-stis/models/min10_max70_PCA'
        gen_model = f.gen_pca_model
    elif model_type == 'svd':
        output_prefix = '/home/amorale4/tf-stis/models/min10_max70_SVD'
        gen_model = f.gen_svd_model
    else:
        print (model_type, 'unknown')
        return

    # if we do them all seperately we dont have to regenerate the tf/b/idf-matrix
    if dt_type == 'tf' or dt_type == 'all':
        update_matrix = True
        for n_topic in k:
            lda_model = gen_model(vectorizer_tf,n_topic, X_train_files, update_mat=update_matrix)
            output = output_prefix+"_Models_tf_n_topics:"+str(n_topic)
            f.save_models(output, lda_model, vectorizer_tf)
            #print("[tffeature]:saved "+str(n_topic)+" k = "+str(k)+" tf", flush=True)
            print("[tffeature]:saved "+str(n_topic)+" k = "+str(k)+" tf")

            if update_matrix:
                update_matrix = False

    if dt_type == 'tfb' or dt_type == 'all':
        update_matrix = True
        for n_topic in k:
            lda_model = gen_model(vectorizer_tfb,n_topic, X_train_files, update_mat=update_matrix)
            output = output_prefix+"_Models_tfb_n_topics:"+str(n_topic)
            f.save_models(output, lda_model, vectorizer_tfb)
            #print("[tffeature]:saved "+str(n_topic)+" k = "+str(k)+" tfb", flush=True)
            print("[tffeature]:saved "+str(n_topic)+" k = "+str(k)+" tfb")

            if update_matrix:
                update_matrix = False

    if dt_type == 'tfidf' or dt_type == 'all':
        update_matrix = True
        for n_topic in k:
            tfv = TopicFeatureVectorizer(useClusters=False, weighting_scheme='unweighted', group_by_keys=['fipscode', 'year'],
                                         merge_on=['fipscode', 'year'])
            output = output_prefix+"_Models_tfidf_n_topics:"+str(n_topic)
            f.save_models(output, lda_model, vectorizer_tfidf)
            #print("[tffeature]:saved "+str(n_topic)+" k = "+str(k)+" tfidf", flush=True)
            print("[tffeature]:saved "+str(n_topic)+" k = "+str(k)+" tfidf")

            if update_matrix:
                update_matrix = False



if __name__=="__main__":
    # gen_model_features(model_type='glda', dt_type='tf')
    # gen_model_features(model_type='glda', dt_type='tfb')
    # gen_model_features(model_type='glda', dt_type='tfidf')

    #gen_model_features(model_type='lda', dt_type='tfidf')
    gen_model_features(model_type='pca', dt_type='tf')
    gen_model_features(model_type='pca', dt_type='tfb')
    gen_model_features(model_type='pca', dt_type='tfidf')

    gen_model_features(model_type='svd', dt_type='tf')
    gen_model_features(model_type='svd', dt_type='tfb')
    gen_model_features(model_type='svd', dt_type='tfidf')
