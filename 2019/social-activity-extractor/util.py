from polyglot.detect import Detector
from konlpy.tag import Okt
import re
import nltk
import sys
from nltk import regexp_tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

okt=Okt()
expression = re.compile('[ㄱ-ㅣ가-힣|a-zA-Z|\s]+')
shortword = re.compile(r'\W*\b\w{1,2}\b')
stop_words = set(stopwords.words('english'))
def process_text(text_data):
	text_data = ''.join(x for x in text_data if x.isprintable())
	text_data = text_data.replace("#", " ")
	text_data = text_data.replace("\n", " ")
	languages = Detector(text_data, quiet=True).languages


	word_list = []
	if languages[0].code in ["ko"]:
		tokens = okt.pos(text_data)
		#print(tokens)
		for token in tokens:
			word = token[0]
			if token[1] in ['Foreign', 'Number', 'URL', 'Email', 'ScreenName', 'Hashtag']:
				# all Hashtag remaining are Japanese
				continue
			elif token[1] == 'Alpha':
				word = word.lower()
			if word == '그램':
				if len(word_list) > 0:
					if word_list[-1] == '스타':
						word_list[-1] = '스타그램'
					elif word_list[-1] == '맛스타':
						word_list[-1] = '맛스타그램'
					else:
						word_list.append(word)
				else:
					word_list.append(word)
			else:
				word_list.append(word)		
	return word_list

def process_text_english(text_data):
	#text_data = shortword.sub('', text_data)
	text_data = ''.join(x for x in text_data if x.isprintable())
	text_data = text_data.replace("#", " ")
	text_data = text_data.replace("\n", " ")
	tokens = okt.pos(text_data)
	word_tokens = word_tokenize(text_data)

	result = []
	for token in tokens:
		word = token[0]
		if token[1] in ['Foreign', 'Number', 'URL', 'Email', 'ScreenName', 'Hashtag']:
			# all Hashtag remaining are Japanese
			continue
		elif token[1] == 'Alpha':
			word = word.lower()
		if word not in stop_words:
			result.append(word)

	return result

def temp(text_data):
	text_data = [re.findall(expression, x) for x in text_data if x.isprintable()]
	word_list = []
	word = []
	for character in text_data:
		if len(character) is 0:
			continue
		if character[0] == ' ' or character[0] == 'ㅤ':
			if len(word) != 0:
				word_list.append(''.join(word))
				word = []
			else:
				continue
		else:
			if character[0].isalpha():
				character[0] = character[0].lower()
			word.append(character[0])
	line = ' '.join(word_list)
