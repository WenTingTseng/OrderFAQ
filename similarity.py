#encoding:utf-8
from __future__ import absolute_import
import jieba
import time
from scipy import spatial
import numpy as np

from load_data import *
#from ckiptagger import WS, POS, NER

file_idf='BM25/idf.txt'

class SSIM(object):
    def __init__(self):
        t1 = time.time()
        self.idf=load_idf(file_idf)
        print("Loading  idf data cost %.3f seconds...\n" % (time.time() - t1))
        #jieba.load_userdict(file_userdict)


    def M_bm25(self,s1, s2, s_avg=5, k1=1.5,k3=2.0 ,b=0.75):
        bm25 = 0
        for w in s1.split():
            idf_s = self.idf.get(w, 1)
            bm25_ra = s2.count(w) * (k1 + 1)
            bm25_rb = s2.count(w) + k1 * ((1 - b) +( b * len(s2))/ s_avg)
            bm25_qa=(k3+1)*s1.count(w)
            bm25_qb=(k3)+s1.count(w)
            bm25 += idf_s * (bm25_ra / bm25_rb)*(bm25_qa/bm25_qb)
        return bm25

    def M_jaccard(self,s1, s2):
        s1 = set(s1)
        s2 = set(s2)
        ret1 = s1.intersection(s2)
        ret2 = s1.union(s2)
        jaccard = 1.0 * len(ret1)/ len(ret2)

        return jaccard

    def ssim(self,s1,s2,model='bm25'):

        if model=='bm25':
            f_ssim=self.M_bm25
        elif model=='jaccard':
            f_ssim=self.M_jaccard

        sim=f_ssim(s1,s2)

        return sim

sm=SSIM()
ssim=sm.ssim