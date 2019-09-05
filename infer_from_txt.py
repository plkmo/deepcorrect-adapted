# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:06:38 2019

@author: tsd
"""
from deepcorrect import corrector_module
import re
from infer import load_model
from argparse import ArgumentParser
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def transcribe(args):
    logger.info("Loading data from file: %s" % args.transcripts)
    with open(args.transcripts, "r", encoding="utf8") as f:
        sents = f.read()
    sents = sents.split('\n')
    
    '''
    if remove_numbers:
        out = []; nums_list = []
        for sent in sents:
            sent1 = re.sub("[0-9]+", "<unk>", sent)
            nums = re.findall("[0-9]+", sent)
            nums_list.append(nums)
            out.append(sent1)
        sents = out
    '''  
    
    logger.info("Loaded %d sentences." % len(sents))
    
    logger.info("Loading model...")
    model_dict = {0: 'deeppunct_checkpoint_tatoeba_cornell',\
                  1: 'deeppunct_checkpoint_google_news',\
                  2: 'deeppunct_checkpoint_wikipedia'}
    corrector = load_model(args.model)
    logger.info("Loaded pre-trained model: %s" % model_dict[args.model])
    
    logger.info("Punctuating...")
    punctuated = []
    for idx, sent in tqdm(enumerate(sents), total=len(sents)):
        try:
            corrected = corrector.correct(sent)[0]['sequence']
            corrected = corrector_module(corrected, sent)
        except Exception as _:
            corrected = sent + "."
            logger.info("Not punctuated due to error: %s" % sent)
        
        punctuated.append(corrected)
        
    logger.info("Punctuated %d sentences." % len(punctuated))
    assert len(sents) == len(punctuated)
    
    logger.info("Writing to file: %s" % args.punctuated)
    with open(args.punctuated, "w", encoding="utf8") as f:
        for p in punctuated:
            f.write("%s\n" % p)
    logger.info("Completed!")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--transcripts", type=str, default="./data/transcripts.txt", \
                        help="path to transcripts file")
    parser.add_argument("--punctuated", type=str, default="./data/punctuated.txt",\
                        help="path to punctuated output file")
    parser.add_argument("--model", type=int, default=2,\
                        help="Choose from 3 pre-trained models:\n\
                        0: deeppunct_checkpoint_tatoeba_cornell\n\
                        1: deeppunct_checkpoint_google_news\n\
                        2: deeppunct_checkpoint_wikipedia")
    args = parser.parse_args()
    
    transcribe(args)
    
        