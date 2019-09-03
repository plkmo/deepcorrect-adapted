# Deepcorrect
(Adapted from https://github.com/bedapudi6788/deepcorrect)

Given unpunctuated (and perhaps also un-capitalized) text, this module helps to punctuate\capitalize the text accordingly. 

## Model Architecture
Seq2Seq LSTM

## Pre-trained models available
0: deeppunct_checkpoint_tatoeba_cornell (trained on tatoeba dataset)  
1: deeppunct_checkpoint_google_news (trained on google news dataset)  
2: deeppunct_checkpoint_wikipedia (trained on Wikipedia dataset)  

## Requirements
Tensorflow=1.14

## Usage
Git clone repo

```bash
git clone https://github.com/plkmo/deepcorrect-adapted
cd ./
infer.py [-h for pre-trained model selection]
```

Output:
```bash
Input sentence to punctuate:
i ate ice cream today the weather is good
09/03/2019 01:49:27 PM [INFO]: Punctuating...
I ate ice cream today, the weather is good.
```

```bash
from infer import load_model
corrector = load_model('./model_data/deeppunct_checkpoint_tatoeba_cornell')
sent = 'I have a dog.'
print(corrector.correct(sent)[0]['sequence'])
```

Output:
```bash
I have a dog.
```

