import json
import nltk 
from nltk.translate.bleu_score import sentence_bleu  
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics import f1_score
class metrics:
    def __init__(self):
        pass
    def calculate_bleu(self,org,ret):


        fac = 1.05
        v = 0
        x = 0
        for i in range(len(org)):

            
            
            nltk_tokens = nltk.word_tokenize(org[i].lower())
            nltk_tokens2 = nltk.word_tokenize(ret[i].lower())

            reference = [nltk_tokens]
              
            
            candidate = nltk_tokens2
            x += sentence_bleu( reference , candidate)
            # print(x)
        print(v)    
#         print(x/len(org))
        return ((x/len(org))/fac)
    def calculate_exact_match_rate(self,org,ret):
        match,total = 0,1e-8
        x = 0 
        for i in range(len(org)):
          if(org[i] == ret[i]):
              match+=1
          else :
              x+=1    
          total += 1    
        print(match/total)
        print(x)
        return match/total
# user = metrics()
# user.calculate_exact_match_rate()    
        
