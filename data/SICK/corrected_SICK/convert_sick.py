
# The script below creates the uncorrected SICK dataset

import pandas as pd
import nltk

test = {'lbls': [], 'hypoths': [], 'prem': []}
train = {'lbls': [], 'hypoths': [], 'prem': []}
dev = {'lbls': [], 'hypoths': [], 'prem': []}
DATA = {"TEST": test, "TRAIN" : train, "TRIAL": dev}

tag_set = set()

line_count = -1
for line in open("../SICK_corrected.tsv"):
  line_count += 1
  if line_count == 0:
    continue
  line = line.split("\t")
  tag = line[-3].strip()
  if tag not in DATA.keys():
    print("Bad tag: %s" % (tag))
  prem = " ".join(nltk.word_tokenize(line[1].strip()))
  hyp = " ".join(nltk.word_tokenize(line[2].strip()))
  lbl = line[-1].lower()
  DATA[tag]['lbls'].append(lbl)
  DATA[tag]['hypoths'].append(hyp)
  DATA[tag]['prem'].append(prem)
  tag_set.add(line[-3])


print(tag_set)

for pair in [("TEST", "test"), ("TRAIN", "train"), ("TRIAL", "dev")]:
  print("Number of %s examples: %d" % (pair[1], len(DATA[pair[0]]['lbls'])))
  lbl_out = open("labels.%s" % (pair[1]), "w")
  prem_out = open("s1.%s" % (pair[1]), "w")
  hyp_out = open("s2.%s" % (pair[1]), "w")
  for i in range(len(DATA[pair[0]]['lbls'])):
    lbl_out.write(DATA[pair[0]]['lbls'][i].strip() + "\n")
    hyp_out.write(DATA[pair[0]]['hypoths'][i].strip() + "\n")
    prem_out.write(DATA[pair[0]]['prem'][i].strip() + "\n")

  hyp_out.close()
  prem_out.close()
  lbl_out.close()
