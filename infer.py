# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 12:49:48 2019

@author: tsd
"""
import pickle
import re
import pandas as pd
import numpy as np
from deepcorrect import DeepCorrect
from argparse import ArgumentParser
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_pickle(filename):
    completeName = filename
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = filename
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def load_model(pretrained):
    logger.info("Loading pre-trained model...")
    model_dict = {0: 'deeppunct_checkpoint_tatoeba_cornell',\
                  1: 'deeppunct_checkpoint_google_news',\
                  2: 'deeppunct_checkpoint_wikipedia'}
    if pretrained not in model_dict.keys():
        model = 'deeppunct_checkpoint_tatoeba_cornell'
    else:
        model = model_dict[pretrained]
    checkpoint_path = "./model_data/%s" % model
    params_path = "./model_data/deeppunct_params_en"
    corrector = DeepCorrect(params_path, checkpoint_path)
    logger.info("Loaded!")
    return corrector

class TED_dataset(object):
    def __init__(self, pretrained=2):
        super(TED_dataset, self).__init__()
        logger.info("Loading processed TED dataset from punc project...")
        self.df = load_pickle("./data/eng.pkl")
        
        def remove_punctuation(sent):
            sent = re.sub(r"[!\?,\.]", " ", sent) # <p>
            return sent
        
        def buffer_punctuation(sent):
            sent = re.sub("\.", " . ", sent)
            sent = re.sub(",", " , ", sent)
            sent = re.sub("!", " ! ", sent)
            sent = re.sub("\?", " ? ", sent)
            return sent
        
        self.df.loc[:, 'sents_nopunc'] = self.df.apply(lambda x: remove_punctuation(x['sents']), axis=1)
        #self.df.loc[:, 'sents'] = self.df.apply(lambda x: buffer_punctuation(x['sents']), axis=1)
        self.corrector = load_model(pretrained)
    
    def buffer_punctuation(self, sent):
        sent = re.sub("\.", " . ", sent)
        sent = re.sub(",", " , ", sent)
        sent = re.sub("!", " ! ", sent)
        sent = re.sub("\?", " ? ", sent)
        return sent
    
    def punctuate_sent(self, sent):
        corrected = self.corrector.correct(sent)[0]['sequence']
        corrected = re.sub("/", ",", corrected) # sub / with comma!!
        return corrected
        
    def punctuate_random(self,):
        corrected = None
        while corrected is None:
            choice = np.random.choice(self.df.index)
            gt = self.df.loc[choice]['sents']
            try:
                corrected = self.corrector.correct(self.df.loc[choice]['sents_nopunc'])[0]['sequence']
                corrected = re.sub("/", ",", corrected) # sub / with comma!!
                print("Punctuated: %s\n" % corrected);
                print("Ground truth: %s" % gt)
            except Exception as _:
                pass
        return corrected, gt
    
    def test_accuracy(self,):
        hits = 0; total = 0; sent_no = 0;
        logger.info("Testing...")
        for idx, (sents, sents_nopunc) in tqdm(enumerate(zip(self.df['sents'], self.df['sents_nopunc'])), total=len(self.df)):
            try:
                corrected = self.corrector.correct(sents_nopunc)[0]['sequence']
                corrected = re.sub("/", ",", corrected) # sub / with comma!!
                corrected = self.buffer_punctuation(corrected)
                sents = self.buffer_punctuation(sents)
                c_labels = re.findall("[\.*\?*\,*!* *]+", corrected)
                s_labels = re.findall("[\.*\?*\,*!* *]+", sents)
                c_labels = [re.sub(r"[ ]+", "", c) for c in c_labels]
                s_labels = [re.sub(r"[ ]+", "", s) for s in s_labels]
                print(c_labels)
                print(s_labels)
                corrected = corrected.split(); sents = sents.split()
                if len(corrected) == len(sents):
                    for c, s in zip(corrected, sents):
                        if s in [".", ",", "!", "?"]:
                            total += 1
                            if s == c:
                                hits += 1
                    sent_no += 1
            except Exception as _:
                continue
            
            if (idx % 10 == 0) and (total > 0):
                print("Punctuated: %s" % corrected)
                print("Ground Truth: %s" % sents)
                print("Sentences considered: %d" % (sent_no))
                print("Total punctuations scored: %d" % total)
                print("Total punctuation hits: %d" % hits)
                print("Accuracy thus far: %.5f %%" % (100*hits/total))

            

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=int, default=2,\
                        help="Choose from 3 pre-trained models:\n\
                        0: deeppunct_checkpoint_tatoeba_cornell\n\
                        1: deeppunct_checkpoint_google_news\n\
                        2: deeppunct_checkpoint_wikipedia")
    args = parser.parse_args()
    
    corrector = load_model(args.model)
    while True:
        sent = input("Input sentence to punctuate:\n")
        if sent in ["quit", "exit"]:
            break
        logger.info("Punctuating...")
        corrected = corrector.correct(sent)[0]['sequence']
        corrected = re.sub("/", ",", corrected) # sub / with comma!!
        print(corrected)