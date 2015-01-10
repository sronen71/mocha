import csv

FILES=["submit6.csv","submit7.csv"]
SUBMIT="ensemble.csv"
PATH="plankton/"
average={}
count=0
for f in FILES:
    with open(PATH+f,'rb') as csvfile:
        count+=1
        print count
        rows=csv.reader(csvfile)
        header=next(rows, None)
        encode=header[1:]
        for k,row in enumerate(rows): 
            predictions=[float(x) for x in row[1:]]
            name=row[0]
            if count==1:
                average[name]=predictions  
            else:
                average[name]=[predictions[i]+average[name][i] 
                        for i in range(0,len(predictions))]
print len(average)
with open(PATH+SUBMIT,'w') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerow(header)
    for key  in average:
        average[key]=[average[key][k]/len(FILES) for k in range(0,len(average[key]))]
        writer.writerow([key]+average[key])
    
