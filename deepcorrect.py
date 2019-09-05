from txt2txt import build_model, infer
import re

def corrector_module(corrected, sent):
    nums = re.findall("[0-9]+", sent)
    corrected = re.sub("/", ",", corrected) # sub / with comma!!
    c_nums = re.findall("[0-9]+", corrected)
    if len(nums) == len(c_nums): # if numbers doesnt match
        for n, c in zip(nums, c_nums):
            if c != n:
                corrected = re.sub(c, n, corrected)
    else: # cant save the unmatched numbers so return original sentence + fullstop
        corrected = sent + "."
    
    if abs(len(corrected.split()) - len(sent.split())) > 3:
        corrected = sent + "."
    
    return corrected

class DeepCorrect():
    deepcorrect_model = None
    def __init__(self, params_path, checkpoint_path):
        # loading the model
        DeepCorrect.deepcorrect_model = build_model(params_path)
        DeepCorrect.deepcorrect_model[0].load_weights(checkpoint_path)
    
    def correct(self, sentence, beam_size = 1):
        #nums = re.findall("[0-9]+", sentence) # added
        if not DeepCorrect.deepcorrect_model:
            print('Please load the model first')

        sentence = sentence.strip()
        sentence = infer(sentence, DeepCorrect.deepcorrect_model[0], DeepCorrect.deepcorrect_model[1], \
                         beam_size = beam_size, max_beams=2, min_cut_off_len=5, cut_off_ratio=1.2)
        
        #sentence = re.sub("/", ",", sentence) # sub / with comma!!
        
        return sentence
