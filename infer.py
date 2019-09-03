# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 12:49:48 2019

@author: tsd
"""

from deepcorrect import DeepCorrect

checkpoint_path = "./model_data/deeppunct_checkpoint_tatoeba_cornell"
params_path = "./model_data/deeppunct_params_en"
corrector = DeepCorrect(params_path, checkpoint_path)
corrector.correct('hey')