from gensim.models.word2vec import Word2Vec
import os
sentences = []

txt_name = "all_func.txt"
f = open('test.txt', 'r')
a = f.read()
apk_name = eval(a)
for i in range(len(apk_name)):
    apk_allFuncs_file = os.path.join(apk_name[i][0], txt_name)
    f2 = open(apk_allFuncs_file, 'r', encoding='utf-8')
    b = f2.read()
    funsList = eval(b)
    sentences.append(funsList)
model= Word2Vec(min_count=5, vector_size=100)
model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
model.save(r'.\fun2vec_model\fun2vecModel\fun2vec')


