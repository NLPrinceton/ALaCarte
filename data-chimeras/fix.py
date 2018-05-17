from string import punctuation
import spacy
import pdb


ERRORS = ['bouries', 'enefies', 'gitbies', 'laccories', 'laffies', 'parinies', 'raffies', 'traxies']
NUMBERS = {str(i) for i in range(10)}
PUNCTUATION = set(punctuation)


with open('dataset.txt', 'r', encoding='latin-1') as f:
  correct = {line[:3]: line for line in f}


for n in [2, 4, 6]:
  print(n)
  for p in ['train', 'test']:
    print(p)
    lines = []
    with open('dataset.l'+str(n)+'.tokenised.'+p+'.txt', 'r', encoding='latin-1') as f:
      for i, line in enumerate(f):
        output = ''
        for error in ERRORS:
          line = line.replace(error, '___').replace('right___', 'right ___').replace('her a ___', 'her an ___').replace('back a ___', 'back an ___').replace('been an ___', 'been a ___').replace('and an ___', 'and a ___').replace('sucking a ___', 'suckin an ___').replace('seen a ___', 'seen an ___').replace('drawing of a ___', 'drawing of an ___').replace('with an ___', 'with a ___').replace('a___', 'a ___')
        key = line[:1]+'_L' if line[1]=='\t' else line[:2]+'_' if line[2]=='\t' else line[:3]
        for i, sentence in enumerate(correct[key].split('@@')[:n]):
          last = ''
          offset = -1
          while line.count(last) > 1:
            prep = ''.join(c for c in sentence.strip().replace("don't", 'do nt').replace("of'em", 'ofem').replace('8pm', '8 pm').replace('E\x85', 'e\x85').replace("'", ' ').replace('"', ' ').replace('-', ' ').split()[offset] if not c in PUNCTUATION)
            offset -= 1
            if not prep:
              continue
            if prep.upper() == prep and not NUMBERS.intersection(prep) and len(prep) > 3 and not prep == 'DRUGS':
              prep = '___'
            last = ' '.join([prep, last]).strip().lower()
          try:
            index = line.index(last.lower())+len(last)
          except:
            pdb.set_trace()
          output += line[:index]+(i<n-1)*' @@'
          line = line[index:]
        output += line
        lines.append(output)
    with open('dataset.l'+str(n)+'.fixed.'+p+'.txt', 'w') as f:
      for line in lines:
        f.write(line)
