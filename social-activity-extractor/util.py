
from polyglot.detect import Detector
from konlpy.tag import Okt
import re

okt=Okt()
expression = re.compile('[ㄱ-ㅣ가-힣|a-zA-Z|\s]+') 
def process_text(text_data):
	text_data = text_data.replace("#", " ")
	text_data = text_data.replace("\n", " ")
	text_data = [re.findall(expression, x) for x in text_data if x.isprintable()]
	#data = [regex.findall('[\p{Hangul}|\p{Latin}|\s]+', x) for x in data if x.isprintable()]
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

	return line