from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

SICK = 0
MSRP = 0
AI 	=  1

if SICK:
	with open("dataset/SICK_train.txt", "r") as f:
		data = f.readlines()

	for i in range(len(data)):
		x = data[i].strip().split('\t')
		pair_id 	= x[0]
		x1 			= x[1]
		x2 			= x[2]
		score 		= x[3]
		judgement 	= x[4]
		stop_words = set(stopwords.words('english')) 
		word_tokens1 = word_tokenize(x1) 
		word_tokens2 = word_tokenize(x2)  
		filtered_sentence1 = [w for w in word_tokens1 if not w in stop_words] 
		filtered_sentence2 = [w for w in word_tokens2 if not w in stop_words]
		y1 = " ".join(filtered_sentence1)
		y2 = " ".join(filtered_sentence2)
		with open("dataset/SICK_train_stop.txt", "a") as f:
			f.write(pair_id + "\t" + y1 + "\t" + y2 + "\t" + score + "\t" + judgement + "\n")

if MSRP:
	with open("dataset/MS_train.txt", "r") as f:
		data = f.readlines()

	for i in range(len(data)):
		x = data[i].strip().split('\t')
		quality = x[0]
		id1 	= x[1]
		id2 	= x[2]
		x1 		= x[3]
		x2 		= x[4]
		stop_words = set(stopwords.words('english')) 
		word_tokens1 = word_tokenize(x1) 
		word_tokens2 = word_tokenize(x2)  
		filtered_sentence1 = [w for w in word_tokens1 if not w in stop_words] 
		filtered_sentence2 = [w for w in word_tokens2 if not w in stop_words]
		y1 = " ".join(filtered_sentence1)
		y2 = " ".join(filtered_sentence2)
		with open("dataset/MS_train_stop.txt", "a") as f:
			f.write(quality + "\t" + id1 + "\t" + id2 + "\t" + y1 + "\t" + y2 + "\n")


if AI:
	with open("dataset/AI_train.txt", "r") as f:
		data = f.readlines()

	for i in range(len(data)):
		x = data[i].strip().split('\t')
		label = x[0]
		x1 		= x[1]
		x2 		= x[2]
		stop_words = set(stopwords.words('english')) 
		word_tokens1 = word_tokenize(x1) 
		word_tokens2 = word_tokenize(x2)  
		filtered_sentence1 = [w for w in word_tokens1 if not w in stop_words] 
		filtered_sentence2 = [w for w in word_tokens2 if not w in stop_words]
		y1 = " ".join(filtered_sentence1)
		y2 = " ".join(filtered_sentence2)
		with open("dataset/AI_train_stop.txt", "a") as f:
			f.write(label + "\t" + y1 + "\t" + y2 + "\n")