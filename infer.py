# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 12:49:48 2019

@author: tsd
"""

from deepcorrect import DeepCorrect
from argparse import ArgumentParser
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_model(pretrained):
    logger.info("Loading pre-trained model...")
    model_dict = {0: 'deeppunct_checkpoint_tatoeba_cornell',\
                  1: 'deeppunct_checkpoint_google_news',\
                  2: 'deeppunct_checkpoint_wikipedia'}
    if pretrained not in model_dict.keys():
        model = 'deeppunct_checkpoint_tatoeba_cornell'
    else:
        model = model_dict[args.model]
    checkpoint_path = "./model_data/%s" % model
    params_path = "./model_data/deeppunct_params_en"
    corrector = DeepCorrect(params_path, checkpoint_path)
    logger.info("Loaded!")
    return corrector

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=int, default=0,\
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
        print(corrector.correct(sent)[0]['sequence'])