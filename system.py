#!/usr/bin/python

from __future__ import division
import numpy as np
from pyemd import emd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from sklearn.metrics import euclidean_distances
from gensim.models import KeyedVectors

# this code is adapted from:
# http://vene.ro/blog/word-movers-distance-in-python.html

def get_wv(lang, wordvectors, binned=False, wvshape=None):
    '''
        Auxiliary function for storing and sorting vocabulary
    '''

    wv = KeyedVectors.load_word2vec_format(wordvectors, binary=binned)

    wv.init_sims(replace=False) # l2 normalizing all wvs
    wvshape = wv.syn0norm.shape

    # saving memmapped file and vocab for posterity
    fp = np.memmap('data/embed_'+lang+'_.dat', dtype=np.double, mode='w+',
            shape=wv.syn0norm.shape)
    fp[:] = wv.syn0norm[:]

    with open('data/embed_'+lang+'_.vocab', 'w') as f:
        for _, w in sorted((voc.index, word) for word, voc in
                wv.vocab.items()):
            print(w, file=f)
    del fp, wv
    # freeing up precious memory

    W = np.memmap('data/embed_'+lang+'_.dat', dtype=np.double, mode='r', shape=wvshape)
    with open('data/embed_'+lang+'_.vocab') as f:
        vocab_list = list(map(str.strip, f.readlines()))

    return W, vocab_list

def objdesc(wvvecs, vocablist, objs, desc):

    vocabdict = {w: k for k, w in enumerate(vocablist)}

    vc = CountVectorizer(stop_words=list(stopwords.words('english'))).fit([objs, desc])

    v_obj, v_desc = vc.transform([objs, desc])

    v_obj = v_obj.toarray().ravel()
    v_desc = v_desc.toarray().ravel()

    wvoc = wvvecs[[vocabdict[w] for w in vc.get_feature_names()]]

    distance_matrix = euclidean_distances(wvoc)

    if np.sum(distance_matrix) == 0.0:
        return float('inf')


    v_obj = v_obj.astype(np.double)
    v_desc = v_desc.astype(np.double)

    v_obj /= v_obj.sum()
    v_desc /= v_desc.sum()

    distance_matrix = distance_matrix.astype(np.double)
    score = emd(v_obj, v_desc, distance_matrix)

    return score


def objdescrefs(wvvecs, vocablist, objs, desc, refs):
    '''
        Function that computes the score given detected objects, description
        and reference list

    '''

    vocabdict = {w: k for k, w in enumerate(vocablist)}

#    objs = 'dog cat cat man'
#    desc = 'a  man with a dog and two cats'
#    refs = ['a man walks with a dog' , 'a cat is walking with a dog']

    vc = CountVectorizer(stop_words='english').fit([objs, desc])


    v_obj, v_desc = vc.transform([objs, desc])

    v_obj = v_obj.toarray().ravel()
    v_desc = v_desc.toarray().ravel()

    wvoc = wvvecs[[vocabdict[w] for w in vc.get_feature_names()]]
    weightsn = np.zeros(len(wvoc))
    for r in refs:
        vr = CountVectorizer(stop_words='english').fit([r])
        wvr = wvvecs[[vocabdict[w] for w in vr.get_feature_names()]]
        wts = (1. - cosine_similarity(wvoc, wvr).max(axis=1))
        wts = np.array([w if np.sign(w) == 1 else 0. for w in wts]) / 2.
        weightsn += wts

    weights = weightsn / len(refs)

    distance_matrix = np.zeros((len(wvoc), len(wvoc)), dtype=np.double)

    for i, o in enumerate(vc.get_feature_names()):
        for j, c in enumerate(vc.get_feature_names()):
            distance_matrix[i,j] = np.sqrt(np.sum(((weights[i] *
                wvvecs[vocabdict[o]]) - (weights[j] *
                    wvvecs[vocabdict[c]]))**2))

    if np.sum(distance_matrix) == 0.0:
        return float('inf')


    v_obj = v_obj.astype(np.double)
    v_desc = v_desc.astype(np.double)

    v_obj /= v_obj.sum()
    v_desc /= v_desc.sum()

    distance_matrix = distance_matrix.astype(np.double)
    # distance_matrix /= distance_matrix.max()
    score = emd(v_obj, v_desc, distance_matrix)

    return score

if __name__ == '__main__':
    import plac
    plac.call(objdescrefs)
