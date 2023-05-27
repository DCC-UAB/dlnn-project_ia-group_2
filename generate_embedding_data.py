import bcolz
import pickle
import numpy as np

main_path_miguel = r'C:\Users\Miguel\OneDrive\Escritorio\dlnn-project_ia-group_2\GLOVE EMBEDDING\glove_files'
main_path_pol = r"C:\Users\polme\OneDrive - UAB\Escritorio\Universitat\Deep Learning\Embeddings"


glove_path = main_path_pol

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.300.dat', mode='w')

with open(f'{glove_path}/glove.6B.300d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(float)
        vectors.append(vect)
    
vectors = bcolz.carray(vectors[1:].reshape((400000, 300)), rootdir=f'{glove_path}/6B.300.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'{glove_path}/6B.300_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'{glove_path}/6B.300_idx.pkl', 'wb'))