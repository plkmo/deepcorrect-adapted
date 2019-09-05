from txt2txt import build_model, infer


class DeepCorrect():
    deepcorrect_model = None
    def __init__(self, params_path, checkpoint_path):
        # loading the model
        DeepCorrect.deepcorrect_model = build_model(params_path)
        DeepCorrect.deepcorrect_model[0].load_weights(checkpoint_path)
    
    def correct(self, sentence, beam_size = 1):
        if not DeepCorrect.deepcorrect_model:
            print('Please load the model first')

        sentence = sentence.strip()
        sentence = infer(sentence, DeepCorrect.deepcorrect_model[0], DeepCorrect.deepcorrect_model[1], \
                         beam_size = beam_size, max_beams=2, min_cut_off_len=5, cut_off_ratio=1.2)
        
        return sentence
