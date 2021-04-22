import random
import torch
import numpy as np
import sys
import os
from torch import optim
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(precision=8)

from toolkits.data_loader import JSONFileDataLoader
from toolkits.framework import FewShotREFramework

# All models
from models.baseline import Base
from models.IA import IA as IA_Model
from models.IA_pair import IAPair as IAPairModel
from models.IADM import IADM as IADM_Model
from models.MANN import MANN as MANN_Model
from models.MAIN import MainModel
from models.MLMAN import MLMAN
from models.PR import PR as PR_Model
from models.PT import PT as PT_Model
from models.Relation import Relation as RelationModel

# All encoders
import toolkits.encoders.encoder_Mascot as MASCOTEncoder
import toolkits.encoders.encoder_CNN as CNNEncoder
import toolkits.encoders.encoder_BRAN as BRANEncoder
import toolkits.encoders.encoder_TE as TransformerEncoder
import argparse

seed = int(np.random.uniform(0, 1)*10000000)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
print('seed: ', seed)


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Level Matching and Aggregation Network for Few-Shot Relation Classification')
    parser.add_argument('--model', type=str, default='base', help='Model name')
    parser.add_argument('--encoder', type=str, default='cnn', help='Encoder name')
    parser.add_argument('--train_iter', type=int, default=50000, help='training iterations')
    parser.add_argument('--N_for_train', type=int, default=10, help='Num of classes for each batch for training')
    parser.add_argument('--N_for_test', type=int, default=5, help='Num of classes for each batch for test')
    parser.add_argument('--K', type=int, default=1, help='Num of instances for each class in the support set')
    parser.add_argument('--Q', type=int, default=5, help='Num of instances for each class in the query set')
    parser.add_argument('--na_rate', type=int, default=0, help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--max_length', type=int, default=40, help='max length of sentence')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument('--load_model', type=str, default=None, help='load pre-trained model')
    parser.add_argument('--save_model', type=str, default='checkpoint/Mascot.tar', help='save trained model path')
    parser.add_argument('--only_test', action='store_true', help='only test')
    parser.add_argument('--test_file', type=str, default='val', help='test file')
    parser.add_argument('--glove', default='pre_trained', help='local or pre_trained')
    parser.add_argument('--reprocess', action='store_true', help='if glove option been changed, reprocess should be True')
    parser.add_argument('--hidden_size', default=60, type=int, help='hidden size')
    parser.add_argument('--optim', type=str, default='SGD', help='optimizer')
    parser.add_argument('--gpu', type=str, default=None)
    args = parser.parse_args()
    print('setting:')
    print(args)

    print("{}-way(train)-{}-way(test)-{}-shot with batch {} Few-Shot Relation Classification"
          .format(args.N_for_train, args.N_for_test, args.K, args.Q))
    print("Model: {}".format(args.model))
    if args.gpu != None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.optim == 'SGD':
        optims = optim.SGD
    elif args.optim == 'Adam':
        optims = optim.Adam

    if not args.only_test:
        test_file = 'val'
    else:
        test_file = args.test_file

    if args.glove == "pre_trained":
        glove_path = './data/glove.6B.50d.json'
    elif args.glove == "local":
        glove_path = './data/glove.local.50d.json'
    else:
        raise RuntimeError("glove option error")

    train_data_loader = JSONFileDataLoader('./data/train.json', glove_path, na_rate=args.na_rate, Q=args.Q, max_length=args.max_length, reprocess=args.reprocess)
    val_data_loader = JSONFileDataLoader('./data/' + test_file + '.json', glove_path, na_rate=args.na_rate, Q=args.Q, max_length=args.max_length, reprocess=args.reprocess)
    test_data_loader = JSONFileDataLoader('./data/' + test_file + '.json', glove_path, na_rate=args.na_rate, Q=args.Q, max_length=args.max_length, reprocess=args.reprocess)

    if args.encoder == "mascot":
        encoder = MASCOTEncoder
    elif args.encoder == "cnn":
        encoder = CNNEncoder
    elif args.encoder == "bran":
        encoder = BRANEncoder
    elif args.encoder == "te":
        encoder = TransformerEncoder
    else:
        raise RuntimeError("incorrect encoder name")

    if args.model == "mascot":
        model = MainModel(train_data_loader.word_vec_mat, args.max_length, hidden_size=args.hidden_size, encoder=encoder, args=args)
    elif args.model == "mlman":
        model = MLMAN(train_data_loader.word_vec_mat, args.max_length, hidden_size=args.hidden_size, encoder=encoder, args=args)
    elif args.model == "base":
        model = Base(train_data_loader.word_vec_mat, args.max_length, hidden_size=args.hidden_size, encoder=encoder, args=args)
    elif args.model == "ia":
        model = IA_Model(train_data_loader.word_vec_mat, args.max_length, hidden_size=args.hidden_size, encoder=encoder, args=args)
    elif args.model == "ia_pair":
        model = IAPairModel(train_data_loader.word_vec_mat, args.max_length, hidden_size=args.hidden_size, encoder=encoder, args=args)
    elif args.model == "iadm":
        model = IADM_Model(train_data_loader.word_vec_mat, args.max_length, hidden_size=args.hidden_size, encoder=encoder, args=args)
    elif args.model == "mann":
        model = MANN_Model(train_data_loader.word_vec_mat, args.max_length, hidden_size=args.hidden_size, encoder=encoder, args=args)
    elif args.model == "pr":
        model = PR_Model(train_data_loader.word_vec_mat, args.max_length, hidden_size=args.hidden_size, encoder=encoder, args=args)
    elif args.model == "pt":
        model = PT_Model(train_data_loader.word_vec_mat, args.max_length, hidden_size=args.hidden_size, encoder=encoder, args=args)
    elif args.model == "relation":
        model = RelationModel(train_data_loader.word_vec_mat, args.max_length, hidden_size=args.hidden_size, encoder=encoder, args=args)
    else:
        raise RuntimeError("incorrect model name")

    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
    if not args.only_test:
        framework.train(
            model, args.batch, N_for_train=args.N_for_train, N_for_eval=args.N_for_test,
            K=args.K, Q=args.Q, learning_rate=args.learning_rate,
            pretrain_model=args.load_model, save_model=args.save_model, optimizer=optims,
            train_iter=args.train_iter, val_iter=1000, val_step=1000, test_iter=2000
        )
    else:
        model = model.cuda()
        test_result = framework.eval(
            model, args.batch, N=args.N_for_test, K=args.K, Q=args.Q, eval_iter=10000, ckpt=args.load_model
        )
        print('test result: {:3.2f}%'.format(test_result*100))


if __name__ == '__main__':
    main()