labels = []

with open('cnews.train.txt', 'r') as f:
    lines = f.readlines()
labels = [line.rstrip('\n').split('\t')[0] for line in lines]

labels = list(set(labels))
with open('labels.txt', 'w') as f:
    f.write('\n'.join(labels))

