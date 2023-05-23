import sys

outTestSetFile = open(sys.argv[2], 'w', encoding='utf-8', errors='ignore')

with open(sys.argv[1], 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        ss = line.split('\t')
        if len(ss) != 4:
            continue
        label, query, title, score = ss

        query = query.lower()
        title = title.lower()

        fLabel = '0'
        if int(label) >= 2:
            fLabel = '1'

        # outTrainSetFile.write(query + '\t' + title + '\t' + score.strip() + '\t' + label + '\n')
        outTestSetFile.write(query + '\t' + title + '\t' + fLabel + '\n')

print('done')