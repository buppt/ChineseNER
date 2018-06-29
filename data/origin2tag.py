# coding=utf-8
import codecs


input_data = codecs.open('origindata.txt','r','utf-8')
output_data = codecs.open('wordtag.txt','w','utf-8')
for line in input_data.readlines():
    line=line.strip()
    i=0
    while i <len(line):
	    if line[i] == '{':
		    i+=2
		    temp=""
		    while line[i]!='}':
			    temp+=line[i]
			    i+=1
		    i+=2
		    word=temp.split(':')
		    sen = word[1]
		    output_data.write(sen[0]+"/B_"+word[0]+" ")
		    for j in sen[1:len(sen)-1]:
			    output_data.write(j+"/M_"+word[0]+" ")
		    output_data.write(sen[-1]+"/E_"+word[0]+" ")
	    else:
		    output_data.write(line[i]+"/O ")
		    i+=1
    output_data.write('\n')
input_data.close()
output_data.close()

