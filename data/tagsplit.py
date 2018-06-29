# coding: utf-8
import re
import codecs

with open('wordtag.txt','rb') as inp:
	texts = inp.read().decode('utf-8')
sentences = re.split('[，。！？、‘’“”（）]/[O]'.decode('utf-8'), texts)
output_data = codecs.open('wordtagsplit.txt','w','utf-8')
for sentence in sentences:
	if sentence != " ":
		output_data.write(sentence.strip()+'\n')
output_data.close()
