from scipy import fft
import random, copy
from random import randint
import csv 
import json 
import time 
import re
import nltk
import string
from numpy import vectorize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import csv 
from nltk.tokenize.treebank import TreebankWordDetokenizer
from my_metrics import metrics
stopwords = stopwords.words('english')
class UserSimulator:
    def __init__(self):
        pass
     
    def control(self):
        db_entity_file = open('/Users/ahmedlokma/Desktop/My_solution_agenda_based/src/Dataset/Guc_Dataset.json','rb')
        dataList= json.load(db_entity_file)
        self.x = []
        self.y = []
        for i in range(len(dataList)):
        #  self.goal = random.randint(0, len(dataList)-1)
         self.goal = i
         dialgoue_intent = self.NLU(self.goal)
         slots = self.nlu_slots(self.goal)
         output_question = self.NLG(slots,self.goal)
         intent_to_agent = self.output_intent(output_question)
         original_ques = self.original_question(self.goal)
         self.x.insert(i,original_ques) # array of original questions for metrics
         self.y.insert(i,output_question) # array of retrieved questions for metrics
        print(len(self.x))
        print(len(dataList)) 
        with open('/Users/ahmedlokma/Desktop/My_solution_agenda_based/src/org1.json', 'w') as f:
         json.dump(self.x, f)
        with open('/Users/ahmedlokma/Desktop/My_solution_agenda_based/src/org22.json', 'w') as f:
         json.dump(self.y, f) 

        newClass = metrics()
        val = newClass.calculate_bleu(self.x,self.y)
        val2 = newClass.calculate_exact_match_rate(self.x,self.y)
        print(val)
        print(val2)

    def NLU(self , index): 

        db_entity_file = open('/Users/ahmedlokma/Desktop/My_solution_agenda_based/src/Dataset/Guc_Dataset.json','rb')
        dataList= json.load(db_entity_file)
        question = dataList[index]
        # print(question['qText'])
        nltk_tokens = nltk.word_tokenize(question['qText'])
        intent = ''
        if('summer' or 'round'  in nltk_tokens):
           intent = 'summer'
        elif ('probation' in nltk_tokens):
           intent = 'probation'
        elif ('advising' or 'advisor' in nltk_tokens):
            intent = 'advising'
        elif ('study' in nltk_tokens):
            intent = 'study_advice'    
        else :
         intent = 'guc_guidelines' 
        return intent  

    def nlu_slots(self,index):  
        db_entity_file2 = open('/Users/ahmedlokma/Desktop/My_solution_agenda_based/src/Dataset/Guc_Dataset_Entity.json','rb')
        dataList2= json.load(db_entity_file2)
        return dataList2[index]['entities']

    def output_intent (self , question):

        db_entity_file2 = open('/Users/ahmedlokma/Desktop/My_solution_agenda_based/src/Dataset/Guc_Dataset.json','rb')
        dataList2= json.load(db_entity_file2)
        intent = ''
        # question = dataList[index]
        for i in range(len(dataList2)-1):
            if(dataList2[i]['qText']== question):
                intent = dataList2[i]['intent']
                return intent 
 
    def NLG(self,slots, index):

        db_entity_file = open('/Users/ahmedlokma/Desktop/My_solution_agenda_based/src/Dataset/Guc_Dataset.json','rb')
        dataList= json.load(db_entity_file)
        db_entity_file2 = open('/Users/ahmedlokma/Desktop/My_solution_agenda_based/src/Dataset/Guc_Dataset_Entity.json','rb')
        dataList2= json.load(db_entity_file2)
        sentences = []
        responses = []
        counter = len(slots)
        original_response = dataList[index]['qText'] 
        max = -1
        for i in  range (len(dataList2)):
        #    original_response = dataList[i]['qText'] 
           counter = len(slots)
        #    max = -1
           for j in dataList2[i]['entities']:
                for z in range(len(slots)):
                  
                  if(slots[z] == j and len(slots) == len(dataList2[i]['entities'])):
                    #  print(j)
                    #  print(slots[z]) 
                     counter = counter -1
                     if(counter > 0):
                      break 
                     elif(counter ==0 ):
                      reterived_response = dataList[i]['qText'] 
                      sentences.insert(0,original_response)
                      sentences.insert(1,reterived_response)
                      cleaned = list(map(self.clean_string,sentences))
                      vectorizer = CountVectorizer().fit_transform(cleaned)
                      vectors = vectorizer.toarray()
                      csim = cosine_similarity(vectors)
                      v = self.cosine_sim_vectors(vectors[0],vectors[1])
                      
                      if(v > max):
                        # print(original_response)
                        # print(reterived_response) 
                        # print('-------------------------') 
                        # print(max)
                        # print(v)
                        max = v
                        responses.insert(0,reterived_response)
                        # print(reterived_response)
                        # print("NLGGGGG")
                        # print(reterived_response)

        # print(original_response)
        # print(responses[0])
        # time.sleep(50)
        return responses[0]    

    def clean_string (self,text) :
        text = ''.join( [ word for word in text if word not in string.punctuation ])
        text = text.lower()
        text = ' '.join([word for word in text.split() if word not in stopwords])
        return text 

    def cosine_sim_vectors (self,vec1 , vec2):
        # print("helllo")
        vec1 = vec1.reshape(1,-1)
        vec2 = vec2.reshape(1,-1)
        return cosine_similarity(vec1 , vec2 )[0][0] 

    def original_question(self,index):
        db_entity_file2 = open('/Users/ahmedlokma/Desktop/My_solution_agenda_based/src/Dataset/Guc_Dataset.json','rb')
        dataList2= json.load(db_entity_file2)
        org_ques = dataList2[index]['qText']
        return org_ques

user = UserSimulator()  
user.control()       
      
