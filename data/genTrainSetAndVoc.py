import sys

outTrainSetFile = open(sys.argv[3], 'w', encoding='utf-8', errors='ignore')

tokenSet = set()
with open(sys.argv[1], 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        ss = line.split('\t')
        if len(ss) != 3:
            continue
        query, title, score = ss

        query = query.lower()
        title = title.lower()

        label = '0'
        if float(score) > 0.7846:
            label = '1'

        # outTrainSetFile.write(query + '\t' + title + '\t' + score.strip() + '\t' + label + '\n')
        outTrainSetFile.write(query + '\t' + title + '\t' + label + '\n')

        for token in query.split():
            tokenSet.add(token)
        for token in title.split():
            tokenSet.add(token)

print('tokenSet size: %d' % len(tokenSet))

vocList = []
vocList.append('<unk>')
vocList.append('<pad>')

for token in tokenSet:
    vocList.append(token)

print('vocList size: %d' % len(vocList))

with open(sys.argv[2], 'w', encoding='utf-8', errors='ignore') as f:
    for token in vocList:
        f.write(token + '\n')

print('done')
