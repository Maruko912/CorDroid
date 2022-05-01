import os
import time
from PIL import Image
import numpy as np
import scipy.sparse as sp
from androguard.core.bytecodes import apk
from androguard.core.bytecodes import dvm
from androguard.core.analysis import analysis

from multiprocessing import Process,Queue

import androguard

import networkx as nx
import matplotlib.pyplot as plt

from gensim.models.word2vec import Word2Vec
# sensitive_apis_path = r"D:\all_code\raw_ICSE_dataset\All8407\8407code\feature_extract\fcg\sensitive_apis.txt"
# model_fun = Word2Vec.load(r'H:\fcg\fun2vec_model\fun2vecModel\fun2vec')
# sen_model_fun = Word2Vec.load(r'H:\fcg\fun2vec_model\senfun2vecModel\fun2vec')
sensitive_apis_path = r"sensitive_apis_drop_crypto.txt"
model_fun = Word2Vec.load(r'.\fun2vec_model\fun2vecModel\fun2vec')
sen_model_fun = Word2Vec.load(r'.\fun2vec_model\senfun2vecModel\fun2vec')


with open(sensitive_apis_path, encoding='utf-8') as f:
    funcs = f.readlines()
    senApis = [func.strip() for func in funcs]

def extract_features(apkfile):

    a = apk.APK(apkfile)
    d = dvm.DalvikVMFormat(a)
    x = analysis.Analysis(d)

    all_func = []
    funcs_call = []
    all_func_index = {}

    # sen_funcs_pair = []
    sen_all_func = []
    sen_funcs_call = []
    sen_all_func_index = {}

    F = np.zeros([256,256])
    M = np.zeros([256,256])

    for method in x.get_methods():
        if method.is_external():
            continue
        m = method.get_method()
        lastcode = None

        caller = str(m).split(' [')[0].split("(")[0]
        if caller not in all_func_index.keys():
            all_func_index[caller] = len(all_func)
            all_func.append(caller)

        for idx, ins in m.get_instructions_idx():
            curcode = ins.get_op_value()
            if(curcode==256):#packed-switch-payload
                curcode = 43 #2b packed-switch
            if(curcode==512):#sparse-switch-payload
                curcode = 44 #2c sparse-switch
            if(curcode==768):#fill-array-data-payload
                curcode = 38 #26 fill-array-data
            if(lastcode != None):
                F[lastcode][curcode] = F[lastcode][curcode] + 1
            lastcode = curcode

            invoke_type = ins.get_name()
            if 'invoke' in invoke_type:
                operands = ins.get_operands()
                callee = operands[-1][-1].split("(")[0]
                if callee not in all_func_index.keys():
                    all_func_index[callee] = len(all_func)
                    all_func.append(callee)
                funcs_call.append([all_func_index[caller],all_func_index[callee]])
                if callee in senApis or caller in senApis:
                    # sen_funcs_pair.append([callee,caller])
                    if callee not in sen_all_func_index.keys():
                        sen_all_func_index[callee] = len(sen_all_func)
                        sen_all_func.append(callee)
                    if caller not in sen_all_func_index.keys():
                        sen_all_func_index[caller] = len(sen_all_func)
                        sen_all_func.append(caller)
                    sen_funcs_call.append([sen_all_func_index[callee], sen_all_func_index[caller]])

    node_features = np.zeros((len(all_func), 100), dtype=np.float32)
    for i in range(len(all_func)):
        # pass
        try:
            node_features[i] = model_fun.wv[all_func[i]]
        except KeyError:
            node_features[i] = [1.0]*100

    sen_node_features = np.zeros((len(sen_all_func), 100), dtype=np.float32)
    for i in range(len(sen_all_func)):
        # pass
        try:
            sen_node_features[i] = sen_model_fun.wv[sen_all_func[i]]
        except KeyError:
            sen_node_features[i] = [1.0]*100
    if len(all_func) == 0:
        print("all_func=0:", apkfile)
    if len(sen_all_func) == 0:
        print("sen_all_func=0:", apkfile)
    adj_calls = np.zeros((len(all_func), len(all_func)))
    for func_calls_pair in funcs_call:
        adj_calls[func_calls_pair[0]][func_calls_pair[1]] = 1

    sen_adj_calls = np.zeros((len(sen_all_func), len(sen_all_func)))
    for sen_func_calls_pair in sen_funcs_call:
        sen_adj_calls[sen_func_calls_pair[0]][sen_func_calls_pair[1]] = 1
    xigemaF =F.sum(axis=1)
    for i in range(256):
        if(xigemaF[i]!=0):
            M[i] = F[i]/xigemaF[i]

    M = M * 255
    img = Image.fromarray(M).convert('L')

    # return all_func, adj_calls, img, sen_all_func, sen_adj_calls

    return all_func, adj_calls, node_features, img, sen_all_func, sen_adj_calls, sen_node_features

def write_features(apk_path, feature_save_dir):

    try:
        all_func, adj_calls, node_features, img, sen_all_func, sen_adj_calls, sen_node_features = extract_features(apk_path)
    # try:
    #     all_func, adj_calls, img, sen_all_func, sen_adj_calls = extract_features(apk_path)
        if not os.path.exists(feature_save_dir):
            os.makedirs(feature_save_dir)
        with open(feature_save_dir + '/all_func.txt', 'w', encoding='utf-8') as f:
            f.write(str(all_func))
        sp.save_npz(feature_save_dir + '/function_call_times.npz', sp.coo_matrix(adj_calls))
        with open(feature_save_dir + '/sen_all_fun.txt', 'w', encoding='utf-8') as f:
            f.write(str(sen_all_func))
        sp.save_npz(feature_save_dir + '/sen_adj_calls.npz', sp.coo_matrix(sen_adj_calls))
        img.save(feature_save_dir + '/opcode_img.png')
        np.save(feature_save_dir + '/node_features.npy', node_features)
        np.save(feature_save_dir + '/sen_node_features.npy', sen_node_features)
    except androguard.core.bytecodes.axml.ResParserError:
        print(apkfile)
        writeFail(apkfile,"androguard.core.bytecodes.axml.ResParserError")
    except AttributeError:
        print(apkfile)
        writeFail(apkfile, "AttributeError")
    except IndexError:
        print(apkfile)
        writeFail(apkfile, "IndexError")
    except TypeError:
        print(apkfile)
        writeFail(apkfile, "TypeError")
    except ValueError:
        print(apkfile)
        writeFail(apkfile,"ValueError-No-dex")

def writeFail(apkfile,Errotype):
    a = [apkfile, Errotype]
    with open('faild_extract.txt', 'a+') as f:
        f.write(str(a)+'\n')
def process_pool(q):
    while True:
        try:
            apk_path, feature_save_dir = q.get(False)
            write_features(apk_path, feature_save_dir)
        except Exception:
            if q.empty():
                break

if __name__=='__main__':

    confused_type_list = [
        "Original",
        #"RandomManifest", "Rebuild", "NewAlignment", "NewSignature",
        # "Rebuild",
        # "MethodRename",
        # "FieldRename", "ClassRename",
        # "LibEncryption", "AssetEncryption", "ConstStringEncryption", "ResStringEncryption",
        # "ArithmeticBranch",
        # "Nop", "Goto",
        # "Reorder",
        # "DebugRemoval",
        # "CallIndirection",
        # "MethodOverload",
        # "Reflection", "AdvancedReflection",
        # "ClassRename_ConstStringEncryption_CallIndirection",
    ]
    for confused_type in confused_type_list:
        start_time = time.time()
        src_root_dir = r'H:\detection\Original'
        # if not os.path.exists(src_root_dir):
        #     continue
        dst_root_dir = r'H:\detection\obfucateAPK_feature'
        all_task = Queue()
        categories = os.listdir(src_root_dir)
        for category in categories:
            apks_dir = os.path.join(src_root_dir, category)
            for apkfile in os.listdir(apks_dir):
                apk_path = os.path.join(apks_dir, apkfile)

                feature_save_dir = os.path.join(dst_root_dir, category, apkfile.split(".")[0])

                temp1 = os.path.join(feature_save_dir,"all_func.txt")
                temp2 = os.path.join(feature_save_dir,"function_call_times.npz")
                temp3 = os.path.join(feature_save_dir,"sen_all_fun.txt")
                temp4 = os.path.join(feature_save_dir,"sen_adj_calls.npz")
                temp5 = os.path.join(feature_save_dir,"opcode_img.png")
                temp6 = os.path.join(feature_save_dir,"sen_node_features.npy")
                temp7 = os.path.join(feature_save_dir,"sen_node_features.npy")
                if os.path.exists(temp1) and os.path.exists(temp2) and os.path.exists(temp3) \
                        and os.path.exists(temp4) and os.path.exists(temp5) and os.path.exists(temp6) and os.path.exists(temp7):
                    continue
                all_task.put((apk_path, feature_save_dir))
        print("process:",confused_type)
        print(all_task.qsize())
        p_list = []
        num_process = 10
        p_list = [Process(target=process_pool, args=(all_task,)) for i in range(num_process)]
        [task.start() for task in p_list]
        [task.join() for task in p_list]
        print("time", time.time() - start_time)

