import os
import time
import multiprocessing
import sys
from gensim.models.word2vec import Word2Vec
import numpy as np


model = Word2Vec.load(r'H:\9133_new\feature_extract\fcg\fun2vec_model\fun2vecModel\fun2vec')


def read_txt_file(funs_path):
    f2 = open(funs_path, 'r', encoding='utf-8')
    # try:
    b = f2.read()
    funsList = eval(b)
    return funsList

def extract_task(dst_output_abs_path):

    all_funs_path = os.path.join(dst_output_abs_path, 'all_func.txt')
    node_features_path = os.path.join(dst_output_abs_path, 'node_features.npy')
    # print(all_funs_path)

    if os.path.exists(all_funs_path):
        all_funcs = read_txt_file(all_funs_path)
        if len(all_funcs)==0:
            print(all_funs_path)
        node_features = np.zeros((len(all_funcs), 100), dtype=np.float32)
        for i in range(len(all_funcs)):
            try:
                node_features[i] = model.wv[all_funcs[i]]
            except:
                node_features[i] = [1.0]*100
        print(dst_output_abs_path)
        np.save(node_features_path, node_features)
    else:
        print("null")

if __name__ == '__main__':

    apk_to_process = []
    root_dir = r'H:\9133_new\9133_data_Original_features'
    pool = multiprocessing.Pool(6)

    for category in os.listdir(root_dir):
        for item in os.listdir(os.path.join(root_dir, category)):
            tmp_path = os.path.join(root_dir, category, item)
            apk_to_process.append(tmp_path)
    pool.map(extract_task, apk_to_process)