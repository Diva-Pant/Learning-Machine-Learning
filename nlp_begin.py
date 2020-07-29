import spacy
nlp = spacy.load("en_core_web_sm")
text = "Learning the Natural Language Processing"
doc = nlp(text)

for token in doc:
	print (token.text, token.lemma_, token.pos, token.is_stop)

#making it more readable
import pandas as pd

cols = ("text","lemma","POS","explain","stopword")
rows = []

for i in doc:
	row = [i.text,i.lemma_,i.pos_,spacy.explain(i.pos_),i.is_stop]
	rows.append(row)
df = pd.DataFrame(rows, columns=cols)

print (df)

#visualizing sparse tree fro the data i.e sentence

from spacy import displacy

#displacy.serve(doc, style="dep")

#sentence segmentation for handling multiple sentences

text = "I am very interested in the field of Machine Learning. I am so amazed at everything this field has to offer. I want to work in the field of Machine Learning. The NLP which is one of the application of ML is quite interesting. It's so much fun."

doc = nlp(text)

for s in doc.sents:
	print(">", s)
# Each sentence has a start and end syntax 

for s in doc.sents:
	print(">",s.start,s.end)

#indexing the document array to get tokens of one/any sentence

print(doc[34:48])
#indexing particular token
token = doc[24]
print(token.text,token.lemma_,token.pos_)

#extracting text from web using Beautiful Soup package

import sys
import warnings
warnings.filterwarnings("ignore")
# Parsing HTML page, getting <p/> tags
from bs4 import BeautifulSoup
import requests
import traceback

def get_text (url):
	buf = []

	try: 
		soup = BeautifulSoup(requests.get(url).text,"html.parser")

		for p in soup.find_all("p"):
			buf.append(p.get_text())
		return "\n".join(buf)
	except:
		print(traceback.format_exc())
		sys.exit(-1)

#getting text online

htm={}
htm["bpk"] = nlp(get_text("https://www.brainpickings.org/"))
htm["kdn"] = nlp(get_text("https://www.kdnuggets.com/2019/09/natural-language-python-using-spacy-introduction.html"))
htm["tds"] = nlp(get_text("https://towardsdatascience.com/intro-to-fastai-installation-and-building-our-first-classifier-938e95fd97d3"))

for s in htm["bpk"].sents:
	print(">", s)

#comparing texts
pairs = [["bpk","kdn"],["bpk","tds"],["kdn","tds"]]
for a,b in pairs:
	print(a,b,htm[a].similarity(htm[b]))
#Natural Language Understanding
#extracting verbs
text = "As Matilda told such dreadful lies, her aunt cannot bear with her and told her friend, who advised her to go to the counsellor. Martha her aunt informed Matilda's father who was stationed in Iraq, as he was an army officer"
doc = nlp(text)
for n in doc.noun_chunks:
	print(n.text)

#finding proper nouns
for pn in doc.ents:
	print(pn.text,pn.label_)
	#displacy.serve(doc,style="ent")
#using WordNet and its spacy integration spacy-wordnet
#using nltk library to download WordNet

import nltk
nltk.download("wordnet")
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
#adding WordnetAnnotator from spacy-wordnet project-- using spaCy pipeline which allows customization
print ("before", nlp.pipe_names)
if "WordnetAnnotator" not in nlp.pipe_names:
	nlp.add_pipe(WordnetAnnotator(nlp.lang), after="tagger")

print("after", nlp.pipe_names)

#words with multiple meanings
token = nlp("date")[0]
print(token._.wordnet.synsets())

#finding their roots
print(token._.wordnet.lemmas())

#finding the domain of the word
print(token._.wordnet.wordnet_domains())