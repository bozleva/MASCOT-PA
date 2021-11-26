import random
import torch
import numpy as np
from sklearn.metrics import f1_score
import sys
import os
import json
import matplotlib.pyplot as plt
from urllib import request
import zipfile


def eval_output():
    from torch import optim
    np.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(precision=8)

    from toolkits.data_loader import ACDFileDataLoader
    from toolkits.framework import FewShotREFramework
    from models.MAIN import MainModel
    from models.baseline import Base
    import toolkits.encoders.encoder_Mascot as MASCOTEncoder
    import toolkits.encoders.encoder_CNN as CNNEncoder
    import argparse

    seed = int(np.random.uniform(0, 1)*10000000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('seed: ', seed)
    # all data store path
    data_path = './data/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mascot', help='Model name')
    parser.add_argument('--N_for_test', type=int, default=5, help='Num of classes for each batch for test')
    parser.add_argument('--K', type=int, default=10, help='Num of instances for each class in the support set')
    parser.add_argument('--Q', type=int, default=1, help='Num of instances for each class in the query set')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--na_rate', type=int, default=0, help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--max_length', type=int, default=40, help='max length of sentence')
    parser.add_argument('--load_model', type=str, default=None, help='load pre-trained model')
    parser.add_argument('--test_file', type=str, default='test_acd', help='test file')
    parser.add_argument('--hidden_size', default=60, type=int, help='hidden size')
    args = parser.parse_args()
    print('setting:')
    print(args)

    print("Model: {}".format(args.model))
    
    test_file = data_path + args.test_file + '.json'
    glove_path = data_path + 'glove.acd.50d.json'

    test_data_loader = ACDFileDataLoader(test_file, glove_path, Q=args.Q, max_length=args.max_length)

    framework = FewShotREFramework(test_data_loader, test_data_loader, test_data_loader)
    if args.model == "mascot":
        model = MainModel(test_data_loader.word_vec_mat, args.max_length, hidden_size=args.hidden_size, encoder=MASCOTEncoder, args=args)
    else:
        model = Base(test_data_loader.word_vec_mat, args.max_length, hidden_size=args.hidden_size, encoder=CNNEncoder, args=args)

    model = model.cuda()
    analyze_result = framework.analyze(
        model, args.batch, N=args.N_for_test, K=args.K, Q=args.Q, eval_iter=10000, ckpt=args.load_model
    )
    with open('log/acd.' + args.model + '.analyze.json', 'w') as fw:
        json.dump(analyze_result, fw)
    print("Finished analyze result!")


def analyze_result(file_path_base, file_path_mascot):

    def gene_datalist(file_path):
        fp = json.load(open(file_path))
        total_len = len(fp['pred'])
        one_size = len(fp['pred'][0])
        pred_list = []
        label_list = []
        class_dict = fp['id2class']
        error_mat = np.zeros([len(class_dict), len(class_dict)])
        for idx in range(total_len):
            trans = {fp['label'][idx][0][l_idx]: fp['id'][idx][0][l_idx] for l_idx in range(one_size)}
            pred_list.extend([trans[p] for p in fp['pred'][idx]])
            label_list.extend([trans[p] for p in fp['label'][idx][0]])

        for i in range(len(pred_list)):
            if pred_list[i] != label_list[i]:
                error_mat[label_list[i], pred_list[i]] += 1
        return label_list, pred_list, error_mat, class_dict
    base_label, base_pred, base_err, class_dict = gene_datalist(file_path_base)
    mascot_label, mascot_pred, mascot_err, class_dict = gene_datalist(file_path_mascot)

    base_fscore = f1_score(base_label, base_pred, average='micro')
    mascot_fscore = f1_score(mascot_label, mascot_pred, average='micro')
    print("base model f1-score: ", base_fscore, "base model f1-score: ", mascot_fscore)

    plt.figure(figsize=(8, 6))

    plt.subplot(121)
    plt.imshow(base_err, cmap='YlOrRd', interpolation='nearest')
    plt.xticks(np.arange(0, len(class_dict)), [class_dict[str(i)] for i in range(len(class_dict))], rotation=45)
    plt.yticks(np.arange(0, len(class_dict)), [class_dict[str(i)] for i in range(len(class_dict))])
    plt.xlabel("base model predict relations")
    plt.ylabel("true relations")

    plt.subplot(122)
    plt.imshow(mascot_err, cmap='YlOrRd', interpolation='nearest')
    plt.xticks(np.arange(0, len(class_dict)), [class_dict[str(i)] for i in range(len(class_dict))], rotation=45)
    plt.yticks(np.arange(0, len(class_dict)), [class_dict[str(i)] for i in range(len(class_dict))])

    plt.xlabel("MASCOT-PA predict relations")
    plt.ylabel("true relations")

    plt.subplots_adjust(bottom=0.06, right=0.95, left=0.15, top=0.9, wspace=0.65)
    cax = plt.axes([0.15, 0.85, 0.75, 0.075])
    plt.colorbar(cax=cax, label="error density", orientation="horizontal")

    plt.savefig("acd_analysis.png")
    # plt.show()

if __name__ == '__main__':
    # eval_output()
    analyze_result('log/acd.base.analyze.json', 'log/acd.mascot.analyze.json')