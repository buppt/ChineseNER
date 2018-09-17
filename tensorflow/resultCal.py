# coding=utf-8
import codecs
def calculate(x,y,id2word,id2tag,res=[]):
    entity=[]
    for i in range(len(x)): #for every sen
        for j in range(len(x[0])): #for every word
            if x[i][j]==0 or y[i][j]==0:
                continue
            if id2tag[y[i][j]][0]=='B':
                entity=[id2word[x[i][j]]+'/'+id2tag[y[i][j]]]
            elif id2tag[y[i][j]][0]=='M' and len(entity)!=0 and entity[-1].split('/')[1][1:]==id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]]+'/'+id2tag[y[i][j]])
            elif id2tag[y[i][j]][0]=='E' and len(entity)!=0 and entity[-1].split('/')[1][1:]==id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]]+'/'+id2tag[y[i][j]])
                entity.append(str(i))
                entity.append(str(j))
                res.append(entity)
                entity=[]
            else:
                entity=[]
    return res
    
    
def calculate3(x,y,id2word,id2tag,res=[]):
    '''
    使用这个函数可以把抽取出的实体写到res.txt文件中，供我们查看。
    注意，这个函数每次使用是在文档的最后添加新信息，所以使用时尽量删除res文件后使用。
    '''
    with codecs.open('./res.txt','a','utf-8') as outp:
        entity=[]
        for i in range(len(x)): #for every sen
            for j in range(len(x[0])): #for every word
                if x[i][j]==0 or y[i][j]==0:
                    continue
                if id2tag[y[i][j]][0]=='B':
                    entity=[id2word[x[i][j]]+'/'+id2tag[y[i][j]]]
                elif id2tag[y[i][j]][0]=='M' and len(entity)!=0 and entity[-1].split('/')[1][1:]==id2tag[y[i][j]][1:]:
                    entity.append(id2word[x[i][j]]+'/'+id2tag[y[i][j]])
                elif id2tag[y[i][j]][0]=='E' and len(entity)!=0 and entity[-1].split('/')[1][1:]==id2tag[y[i][j]][1:]:
                    entity.append(id2word[x[i][j]]+'/'+id2tag[y[i][j]])
                    entity.append(str(i))
                    entity.append(str(j))
                    res.append(entity)
                    st = ""
                    for s in entity:
                        st += s+' '
                    #print st
                    outp.write(st+'\n')
                    entity=[]
                else:
                    entity=[]
    return res
