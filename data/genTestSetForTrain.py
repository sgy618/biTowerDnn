import sys
import random

outTrainSetFile = open(sys.argv[2], 'w', encoding='utf-8', errors='ignore')

sampleList = []
with open(sys.argv[1], 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        ss = line.split('\t')
        if len(ss) != 3:
            continue
        query, title, score = ss

        query = query.lower()
        title = title.lower()

        label = '0'
        if float(score.strip()) > 0.7846:
            label = '1'

        sampleList.append((query, title, label))

print('sampleList size: %d' % len(sampleList))

random.shuffle(sampleList)

for sample in sampleList[:10000]:
    query, title, label = sample
    outTrainSetFile.write(query + '\t' + title + '\t' + label + '\n')