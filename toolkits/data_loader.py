import json
import os
import numpy as np
import random
import torch


class JSONFileDataLoader(object):

    def __init__(self, file_name, word_vec_file_name, max_length=40, na_rate=0, Q=1, case_sensitive=False, reprocess=False):
        self.file_name = file_name
        self.word_vec_file_name = word_vec_file_name
        self.case_sensitive = case_sensitive
        self.max_length = max_length
        self.Q = Q
        self.na_rate = na_rate

        if reprocess or not self._load_preprocessed_file():  # Try to load pre-processed files:
            # Check files
            if file_name is None or not os.path.isfile(file_name):
                raise Exception("[ERROR] Data file doesn't exist")
            if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
                raise Exception("[ERROR] Word vector file doesn't exist")

            # Load files
            print("Loading data file...", self.file_name)
            self.ori_data = json.load(open(self.file_name, "r"))
            print("Finish loading")
            print("Loading word vector file...")
            self.ori_word_vec = json.load(open(self.word_vec_file_name, "r"))
            print("Finish loading")

            # Eliminate case sensitive
            if not case_sensitive:
                print("Elimiating case sensitive problem...")
                for relation in self.ori_data:
                    for ins in self.ori_data[relation]:
                        for i in range(len(ins['tokens'])):
                            ins['tokens'][i] = ins['tokens'][i].lower()
                print("Finish eliminating")

            # Pre-process word vec
            self.word2id = {}
            self.word_vec_tot = len(self.ori_word_vec)
            UNK = self.word_vec_tot
            PAD = self.word_vec_tot + 1
            self.word_vec_dim = len(self.ori_word_vec[0]['vec'])
            print("Got {} words of {} dims".format(self.word_vec_tot, self.word_vec_dim))
            print("Building word vector matrix and mapping...")
            self.word_vec_mat = np.zeros((self.word_vec_tot, self.word_vec_dim), dtype=np.float32)
            for cur_id, word in enumerate(self.ori_word_vec):
                w = word['word']
                if not case_sensitive:
                    w = w.lower()
                self.word2id[w] = cur_id
                self.word_vec_mat[cur_id, :] = word['vec']
                self.word_vec_mat[cur_id] = self.word_vec_mat[cur_id] / np.sqrt(np.sum(self.word_vec_mat[cur_id] ** 2))
            self.word2id['UNK'] = UNK
            self.word2id['PAD'] = PAD
            print("Finish building")

            # Pre-process data
            print("Pre-processing data...")
            self.instance_tot = 0
            for relation in self.ori_data:
                self.instance_tot += len(self.ori_data[relation])
            self.data_word = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos1 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos2 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_mask = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_length = np.zeros((self.instance_tot), dtype=np.int32)
            self.rel2scope = {}  # left closed and right open
            i = 0
            for relation in self.ori_data:
                self.rel2scope[relation] = [i, i]
                for ins in self.ori_data[relation]:
                    pos1 = ins['h'][2][0][0]
                    pos2 = ins['t'][2][0][0]
                    words = ins['tokens']
                    cur_ref_data_word = self.data_word[i]
                    for j, word in enumerate(words):
                        if j < max_length:
                            if word in self.word2id:
                                cur_ref_data_word[j] = self.word2id[word]
                            else:
                                cur_ref_data_word[j] = UNK
                        else:
                            break
                    for j in range(j + 1, max_length):
                        cur_ref_data_word[j] = PAD
                    self.data_length[i] = len(words)
                    if len(words) > max_length:
                        self.data_length[i] = max_length
                    if pos1 >= max_length:
                        pos1 = max_length - 1
                    if pos2 >= max_length:
                        pos2 = max_length - 1
                    pos_min = min(pos1, pos2)
                    pos_max = max(pos1, pos2)
                    for j in range(max_length):
                        self.data_pos1[i][j] = j - pos1 + max_length
                        self.data_pos2[i][j] = j - pos2 + max_length
                        if j >= self.data_length[i]:
                            self.data_mask[i][j] = 0
                            self.data_pos1[i][j] = 0
                            self.data_pos2[i][j] = 0
                        elif j <= pos_min:
                            self.data_mask[i][j] = 1
                        elif j <= pos_max:
                            self.data_mask[i][j] = 2
                        else:
                            self.data_mask[i][j] = 3
                    i += 1
                self.rel2scope[relation][1] = i

            print("Finish pre-processing")
            print("Storing processed files...")
            name_prefix = '.'.join(file_name.split('/')[-1].split('.')[:-1])
            word_vec_name_prefix = '.'.join(word_vec_file_name.split('/')[-1].split('.')[:-1])
            processed_data_dir = '_processed_data'
            if not os.path.isdir(processed_data_dir):
                os.mkdir(processed_data_dir)
            np.save(os.path.join(processed_data_dir, name_prefix + '_word.npy'), self.data_word)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos1.npy'), self.data_pos1)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos2.npy'), self.data_pos2)
            np.save(os.path.join(processed_data_dir, name_prefix + '_mask.npy'), self.data_mask)
            np.save(os.path.join(processed_data_dir, name_prefix + '_length.npy'), self.data_length)
            json.dump(self.rel2scope, open(os.path.join(processed_data_dir, name_prefix + '_rel2scope.json'), 'w'))
            np.save(os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy'), self.word_vec_mat)
            json.dump(self.word2id, open(os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json'), 'w'))
            print("Finish storing")

    def _load_preprocessed_file(self):
        name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
        word_vec_name_prefix = '.'.join(self.word_vec_file_name.split('/')[-1].split('.')[:-1])
        processed_data_dir = '_processed_data'
        if not os.path.isdir(processed_data_dir):
            return False
        word_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_word.npy')
        pos1_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos1.npy')
        pos2_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos2.npy')
        mask_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_mask.npy')
        length_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_length.npy')
        rel2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2scope.json')
        word_vec_mat_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy')
        word2id_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json')
        if not os.path.exists(word_npy_file_name) or \
           not os.path.exists(pos1_npy_file_name) or \
           not os.path.exists(pos2_npy_file_name) or \
           not os.path.exists(mask_npy_file_name) or \
           not os.path.exists(length_npy_file_name) or \
           not os.path.exists(rel2scope_file_name) or \
           not os.path.exists(word_vec_mat_file_name) or \
           not os.path.exists(word2id_file_name):
            return False
        print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word_npy_file_name)
        self.data_pos1 = np.load(pos1_npy_file_name)
        self.data_pos2 = np.load(pos2_npy_file_name)
        self.data_mask = np.load(mask_npy_file_name)
        self.data_length = np.load(length_npy_file_name)
        self.rel2scope = json.load(open(rel2scope_file_name))
        self.word_vec_mat = np.load(word_vec_mat_file_name)
        self.word2id = json.load(open(word2id_file_name))
        if self.data_word.shape[1] != self.max_length:
            print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        print("Finish loading")
        return True

    def next_one(self, N=5, K=5, Q=100):
        target_classes = random.sample(self.rel2scope.keys(), N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes, self.rel2scope.keys()))
        for i, class_name in enumerate(target_classes):
            scope = self.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), K + Q, False)
            word = self.data_word[indices]
            pos1 = self.data_pos1[indices]
            pos2 = self.data_pos2[indices]
            mask = self.data_mask[indices]
            support_word, query_word = np.split(word, [K])
            support_pos1, query_pos1 = np.split(pos1, [K])
            support_pos2, query_pos2 = np.split(pos2, [K])
            support_mask, query_mask = np.split(mask, [K])
            support_set['word'].append(support_word)
            support_set['pos1'].append(support_pos1)
            support_set['pos2'].append(support_pos2)
            support_set['mask'].append(support_mask)
            query_set['word'].append(query_word)
            query_set['pos1'].append(query_pos1)
            query_set['pos2'].append(query_pos2)
            query_set['mask'].append(query_mask)
            query_label += [i] * Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            scope = self.rel2scope[cur_class]
            indices = np.random.choice(list(range(scope[0], scope[1])), Q, False)
            word = self.data_word[indices]
            pos1 = self.data_pos1[indices]
            pos2 = self.data_pos2[indices]
            mask = self.data_mask[indices]
            query_word = [q[0] for q in np.split(word, Q)]
            query_pos1 = [q[0] for q in np.split(pos1, Q)]
            query_pos2 = [q[0] for q in np.split(pos2, Q)]
            query_mask = [q[0] for q in np.split(mask, Q)]
            query_set['word'].append(query_word)
            query_set['pos1'].append(query_pos1)
            query_set['pos2'].append(query_pos2)
            query_set['mask'].append(query_mask)
        query_label += [N] * Q_na

        support_set['word'] = np.stack(support_set['word'], 0)
        support_set['pos1'] = np.stack(support_set['pos1'], 0)
        support_set['pos2'] = np.stack(support_set['pos2'], 0)
        support_set['mask'] = np.stack(support_set['mask'], 0)
        query_set['word'] = np.concatenate(query_set['word'], 0)
        query_set['pos1'] = np.concatenate(query_set['pos1'], 0)
        query_set['pos2'] = np.concatenate(query_set['pos2'], 0)
        query_set['mask'] = np.concatenate(query_set['mask'], 0)
        query_label = np.array(query_label)

        return support_set, query_set, query_label

    def next_batch(self, B=4, N=20, K=5, Q=100):
        support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        label = []
        for one_sample in range(B):

            current_support, current_query, current_label = self.next_one(N, K, Q)

            index = [i for i in range(len(current_label))]
            random.shuffle(index)

            support['word'].append(current_support['word'])
            support['pos1'].append(current_support['pos1'])
            support['pos2'].append(current_support['pos2'])
            support['mask'].append(current_support['mask'])
            query['word'].append(current_query['word'][index])
            query['pos1'].append(current_query['pos1'][index])
            query['pos2'].append(current_query['pos2'][index])
            query['mask'].append(current_query['mask'][index])
            label.append(current_label[index])

        support['word'] = torch.from_numpy(np.stack(support['word'], 0)).long().view(-1, self.max_length)
        support['pos1'] = torch.from_numpy(np.stack(support['pos1'], 0)).long().view(-1, self.max_length)
        support['pos2'] = torch.from_numpy(np.stack(support['pos2'], 0)).long().view(-1, self.max_length)
        support['mask'] = torch.from_numpy(np.stack(support['mask'], 0)).long().view(-1, self.max_length)
        query['word'] = torch.from_numpy(np.stack(query['word'], 0)).long().view(-1, self.max_length)
        query['pos1'] = torch.from_numpy(np.stack(query['pos1'], 0)).long().view(-1, self.max_length)
        query['pos2'] = torch.from_numpy(np.stack(query['pos2'], 0)).long().view(-1, self.max_length)
        query['mask'] = torch.from_numpy(np.stack(query['mask'], 0)).long().view(-1, self.max_length)
        label = torch.from_numpy(np.stack(label, 0).astype(np.int64)).long()

        for key in support:
            support[key] = support[key].cuda()
        for key in query:
            query[key] = query[key].cuda()
        label = label.cuda()

        return support, query, label

class ACDFileDataLoader(object):

    def __init__(self, file_name, word_vec_file_name, max_length=40, na_rate=0, Q=1, case_sensitive=False, reprocess=False):
        self.file_name = file_name
        self.word_vec_file_name = word_vec_file_name
        self.case_sensitive = case_sensitive
        self.max_length = max_length
        self.Q = Q
        self.na_rate = na_rate
        self.class_id_dict = {k: c_id for c_id, k in enumerate(json.load(open(self.file_name, "r")).keys())}
        self.id_class_dict = {c_id: k for k, c_id in self.class_id_dict.items()}

        if reprocess or not self._load_preprocessed_file():  # Try to load pre-processed files:
            # Check files
            if file_name is None or not os.path.isfile(file_name):
                raise Exception("[ERROR] Data file doesn't exist")
            if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
                raise Exception("[ERROR] Word vector file doesn't exist")

            # Load files
            print("Loading data file...")
            self.ori_data = json.load(open(self.file_name, "r"))
            print("Finish loading")
            print("Loading word vector file...")
            self.ori_word_vec = json.load(open(self.word_vec_file_name, "r"))
            print("Finish loading")

            # Eliminate case sensitive
            if not case_sensitive:
                print("Elimiating case sensitive problem...")
                for relation in self.ori_data:
                    for ins in self.ori_data[relation]:
                        for i in range(len(ins['tokens'])):
                            ins['tokens'][i] = ins['tokens'][i].lower()
                print("Finish eliminating")

            # Pre-process word vec
            self.word2id = {}
            self.word_vec_tot = len(self.ori_word_vec)
            UNK = self.word_vec_tot
            PAD = self.word_vec_tot + 1
            self.word_vec_dim = len(self.ori_word_vec[0]['vec'])
            print("Got {} words of {} dims".format(self.word_vec_tot, self.word_vec_dim))
            print("Building word vector matrix and mapping...")
            self.word_vec_mat = np.zeros((self.word_vec_tot, self.word_vec_dim), dtype=np.float32)
            for cur_id, word in enumerate(self.ori_word_vec):
                w = word['word']
                if not case_sensitive:
                    w = w.lower()
                self.word2id[w] = cur_id
                self.word_vec_mat[cur_id, :] = word['vec']
                self.word_vec_mat[cur_id] = self.word_vec_mat[cur_id] / np.sqrt(np.sum(self.word_vec_mat[cur_id] ** 2))
            self.word2id['UNK'] = UNK
            self.word2id['PAD'] = PAD
            print("Finish building")

            # Pre-process data
            print("Pre-processing data...")
            self.instance_tot = 0
            for relation in self.ori_data:
                self.instance_tot += len(self.ori_data[relation])
            self.data_word = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos1 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos2 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_mask = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_length = np.zeros((self.instance_tot), dtype=np.int32)
            self.rel2scope = {}  # left closed and right open
            i = 0
            for relation in self.ori_data:
                self.rel2scope[relation] = [i, i]
                for ins in self.ori_data[relation]:
                    pos1 = ins['h'][2][0][0]
                    pos2 = ins['t'][2][0][0]
                    words = ins['tokens']
                    cur_ref_data_word = self.data_word[i]
                    for j, word in enumerate(words):
                        if j < max_length:
                            if word in self.word2id:
                                cur_ref_data_word[j] = self.word2id[word]
                            else:
                                cur_ref_data_word[j] = UNK
                        else:
                            break
                    for j in range(j + 1, max_length):
                        cur_ref_data_word[j] = PAD
                    self.data_length[i] = len(words)
                    if len(words) > max_length:
                        self.data_length[i] = max_length
                    if pos1 >= max_length:
                        pos1 = max_length - 1
                    if pos2 >= max_length:
                        pos2 = max_length - 1
                    pos_min = min(pos1, pos2)
                    pos_max = max(pos1, pos2)
                    for j in range(max_length):
                        self.data_pos1[i][j] = j - pos1 + max_length
                        self.data_pos2[i][j] = j - pos2 + max_length
                        if j >= self.data_length[i]:
                            self.data_mask[i][j] = 0
                            self.data_pos1[i][j] = 0
                            self.data_pos2[i][j] = 0
                        elif j <= pos_min:
                            self.data_mask[i][j] = 1
                        elif j <= pos_max:
                            self.data_mask[i][j] = 2
                        else:
                            self.data_mask[i][j] = 3
                    i += 1
                self.rel2scope[relation][1] = i

            print("Finish pre-processing")
            print("Storing processed files...")
            name_prefix = '.'.join(file_name.split('/')[-1].split('.')[:-1])
            word_vec_name_prefix = '.'.join(word_vec_file_name.split('/')[-1].split('.')[:-1])
            processed_data_dir = '_processed_data'
            if not os.path.isdir(processed_data_dir):
                os.mkdir(processed_data_dir)
            np.save(os.path.join(processed_data_dir, name_prefix + '_word.npy'), self.data_word)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos1.npy'), self.data_pos1)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos2.npy'), self.data_pos2)
            np.save(os.path.join(processed_data_dir, name_prefix + '_mask.npy'), self.data_mask)
            np.save(os.path.join(processed_data_dir, name_prefix + '_length.npy'), self.data_length)
            json.dump(self.rel2scope, open(os.path.join(processed_data_dir, name_prefix + '_rel2scope.json'), 'w'))
            np.save(os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy'), self.word_vec_mat)
            json.dump(self.word2id, open(os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json'), 'w'))
            print("Finish storing")

    def _load_preprocessed_file(self):
        name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
        word_vec_name_prefix = '.'.join(self.word_vec_file_name.split('/')[-1].split('.')[:-1])
        processed_data_dir = '_processed_data'
        if not os.path.isdir(processed_data_dir):
            return False
        word_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_word.npy')
        pos1_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos1.npy')
        pos2_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos2.npy')
        mask_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_mask.npy')
        length_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_length.npy')
        rel2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2scope.json')
        word_vec_mat_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy')
        word2id_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json')
        if not os.path.exists(word_npy_file_name) or \
           not os.path.exists(pos1_npy_file_name) or \
           not os.path.exists(pos2_npy_file_name) or \
           not os.path.exists(mask_npy_file_name) or \
           not os.path.exists(length_npy_file_name) or \
           not os.path.exists(rel2scope_file_name) or \
           not os.path.exists(word_vec_mat_file_name) or \
           not os.path.exists(word2id_file_name):
            return False
        print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word_npy_file_name)
        self.data_pos1 = np.load(pos1_npy_file_name)
        self.data_pos2 = np.load(pos2_npy_file_name)
        self.data_mask = np.load(mask_npy_file_name)
        self.data_length = np.load(length_npy_file_name)
        self.rel2scope = json.load(open(rel2scope_file_name))
        self.word_vec_mat = np.load(word_vec_mat_file_name)
        self.word2id = json.load(open(word2id_file_name))
        if self.data_word.shape[1] != self.max_length:
            print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        print("Finish loading")
        return True

    def next_one(self, N=5, K=5, Q=100):
        target_classes = random.sample(self.rel2scope.keys(), N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_label = []
        query_class = []
        for i, class_name in enumerate(target_classes):
            scope = self.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), K + Q, False)
            word = self.data_word[indices]
            pos1 = self.data_pos1[indices]
            pos2 = self.data_pos2[indices]
            mask = self.data_mask[indices]
            support_word, query_word = np.split(word, [K])
            support_pos1, query_pos1 = np.split(pos1, [K])
            support_pos2, query_pos2 = np.split(pos2, [K])
            support_mask, query_mask = np.split(mask, [K])
            support_set['word'].append(support_word)
            support_set['pos1'].append(support_pos1)
            support_set['pos2'].append(support_pos2)
            support_set['mask'].append(support_mask)
            query_set['word'].append(query_word)
            query_set['pos1'].append(query_pos1)
            query_set['pos2'].append(query_pos2)
            query_set['mask'].append(query_mask)
            query_label += [i] * Q
            query_class += [self.class_id_dict[class_name]] * Q

        support_set['word'] = np.stack(support_set['word'], 0)
        support_set['pos1'] = np.stack(support_set['pos1'], 0)
        support_set['pos2'] = np.stack(support_set['pos2'], 0)
        support_set['mask'] = np.stack(support_set['mask'], 0)
        query_set['word'] = np.concatenate(query_set['word'], 0)
        query_set['pos1'] = np.concatenate(query_set['pos1'], 0)
        query_set['pos2'] = np.concatenate(query_set['pos2'], 0)
        query_set['mask'] = np.concatenate(query_set['mask'], 0)
        query_label = np.array(query_label)
        query_class = np.array(query_class)

        return support_set, query_set, query_label, query_class

    def next_batch(self, B=4, N=20, K=5, Q=100):
        support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        label = []
        class_id = []
        for one_sample in range(B):

            current_support, current_query, current_label, current_class = self.next_one(N, K, Q)

            index = [i for i in range(len(current_label))]
            random.shuffle(index)

            support['word'].append(current_support['word'])
            support['pos1'].append(current_support['pos1'])
            support['pos2'].append(current_support['pos2'])
            support['mask'].append(current_support['mask'])
            query['word'].append(current_query['word'][index])
            query['pos1'].append(current_query['pos1'][index])
            query['pos2'].append(current_query['pos2'][index])
            query['mask'].append(current_query['mask'][index])
            label.append(current_label[index])
            class_id.append(current_class[index])

        support['word'] = torch.from_numpy(np.stack(support['word'], 0)).long().view(-1, self.max_length)
        support['pos1'] = torch.from_numpy(np.stack(support['pos1'], 0)).long().view(-1, self.max_length)
        support['pos2'] = torch.from_numpy(np.stack(support['pos2'], 0)).long().view(-1, self.max_length)
        support['mask'] = torch.from_numpy(np.stack(support['mask'], 0)).long().view(-1, self.max_length)
        query['word'] = torch.from_numpy(np.stack(query['word'], 0)).long().view(-1, self.max_length)
        query['pos1'] = torch.from_numpy(np.stack(query['pos1'], 0)).long().view(-1, self.max_length)
        query['pos2'] = torch.from_numpy(np.stack(query['pos2'], 0)).long().view(-1, self.max_length)
        query['mask'] = torch.from_numpy(np.stack(query['mask'], 0)).long().view(-1, self.max_length)
        label = torch.from_numpy(np.stack(label, 0).astype(np.int64)).long()
        class_id = torch.from_numpy(np.stack(class_id, 0).astype(np.int64)).long()

        for key in support:
            support[key] = support[key].cuda()
        for key in query:
            query[key] = query[key].cuda()
        label = label.cuda()
        class_id = class_id.cuda()

        return support, query, label, class_id

class JSONTestFileDataLoader(object):

    def __init__(self, test_data_path, file_root_name, word_vec_file_name, max_length=40, args=None):
        self.test_data_path = test_data_path
        self.file_name = os.path.join(test_data_path, '-'.join([file_root_name, str(args.N_for_test), str(args.K)]) + '.json')
        self.json_data = json.load(open(self.file_name, 'r', encoding='utf8'))
        self.word_vec_file_name = word_vec_file_name
        self.max_length = max_length
        self.N = args.N_for_test
        self.K = args.K
        self.Q = args.Q
        self.na_rate = args.na_rate
        self.total_len = len(self.json_data)

        self.word2vec = {pair['word']: pair['vec'] for pair in json.load(open(self.word_vec_file_name, 'r', encoding='utf8'))}
        self.word2id = {pair['word']: idx for idx, pair in enumerate(json.load(open(self.word_vec_file_name, 'r', encoding='utf8')))}
        self.word_vec_file_name = word_vec_file_name
        self.ori_word_vec = json.load(open(self.word_vec_file_name, "r"))

        self.max_length = max_length
        self.word2id = {}
        self.word_vec_tot = len(self.ori_word_vec)
        UNK = self.word_vec_tot
        PAD = self.word_vec_tot + 1
        self.word2id['UNK'] = UNK
        self.word2id['PAD'] = PAD
        if not self._load_preprocessed_file():

            self.word_vec_dim = len(self.ori_word_vec[0]['vec'])
            print("Got {} words of {} dims".format(self.word_vec_tot, self.word_vec_dim))
            print("Building word vector matrix and mapping...")
            self.word_vec_mat = np.zeros((self.word_vec_tot, self.word_vec_dim), dtype=np.float32)
            for cur_id, word in enumerate(self.ori_word_vec):
                w = word['word']
                w = w.lower()
                self.word2id[w] = cur_id
                self.word_vec_mat[cur_id, :] = word['vec']
                self.word_vec_mat[cur_id] = self.word_vec_mat[cur_id] / np.sqrt(np.sum(self.word_vec_mat[cur_id] ** 2))


    def _load_preprocessed_file(self):
        name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
        # name_prefix = 'train'
        word_vec_name_prefix = '.'.join(self.word_vec_file_name.split('/')[-1].split('.')[:-1])
        processed_data_dir = '_processed_data'
        if not os.path.isdir(processed_data_dir):
            return False
        word_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_word.npy')
        pos1_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos1.npy')
        pos2_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos2.npy')
        mask_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_mask.npy')
        length_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_length.npy')
        rel2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2scope.json')
        word_vec_mat_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy')
        word2id_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json')
        if not os.path.exists(word_npy_file_name) or \
                not os.path.exists(pos1_npy_file_name) or \
                not os.path.exists(pos2_npy_file_name) or \
                not os.path.exists(mask_npy_file_name) or \
                not os.path.exists(length_npy_file_name) or \
                not os.path.exists(rel2scope_file_name) or \
                not os.path.exists(word_vec_mat_file_name) or \
                not os.path.exists(word2id_file_name):
            return False
        print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word_npy_file_name)
        self.data_pos1 = np.load(pos1_npy_file_name)
        self.data_pos2 = np.load(pos2_npy_file_name)
        self.data_mask = np.load(mask_npy_file_name)
        self.data_length = np.load(length_npy_file_name)
        self.rel2scope = json.load(open(rel2scope_file_name))
        self.word_vec_mat = np.load(word_vec_mat_file_name)
        self.word2id = json.load(open(word2id_file_name))
        if self.data_word.shape[1] != self.max_length:
            print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        print("Finish loading")
        return True

    def get_std_data(self, ins):
        data_pos1 = np.zeros((self.max_length), dtype=np.int32)
        data_pos2 = np.zeros((self.max_length), dtype=np.int32)
        data_mask = np.zeros((self.max_length), dtype=np.int32)
        cur_ref_data_word = np.zeros((self.max_length), dtype=np.int32)
        pos1 = ins['h'][2][0][0]
        pos2 = ins['t'][2][0][0]
        words = ins['tokens']
        data_length = len(words)
        for j, word in enumerate(words):
            if j < self.max_length:
                if word in self.word2id:
                    cur_ref_data_word[j] = self.word2id[word]
                else:
                    cur_ref_data_word[j] = self.word2id['UNK']
            else:
                break
        for j in range(j + 1, self.max_length):
            cur_ref_data_word[j] = self.word2id['PAD']
        if pos1 >= self.max_length:
            pos1 = self.max_length - 1
        if pos2 >= self.max_length:
            pos2 = self.max_length - 1
        pos_min = min(pos1, pos2)
        pos_max = max(pos1, pos2)
        for j in range(self.max_length):
            data_pos1[j] = j - pos1 + self.max_length
            data_pos2[j] = j - pos2 + self.max_length
            if j >= data_length:
                data_mask[j] = 0
                data_pos1[j] = 0
                data_pos2[j] = 0
            elif j <= pos_min:
                data_mask[j] = 1
            elif j <= pos_max:
                data_mask[j] = 2
            else:
                data_mask[j] = 3

        return cur_ref_data_word, data_pos1, data_pos2, data_mask

    def next_one(self, it=0, N=5, K=5, Q=100):
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_label = []

        for s_ins in self.json_data[it]['meta_train']:
            support_word = []
            support_pos1 = []
            support_pos2 = []
            support_mask = []
            for ins in s_ins:
                word, pos1, pos2, mask = self.get_std_data(ins)
                support_word.append(word)
                support_pos1.append(pos1)
                support_pos2.append(pos2)
                support_mask.append(mask)
            support_set['word'].append(support_word)
            support_set['pos1'].append(support_pos1)
            support_set['pos2'].append(support_pos2)
            support_set['mask'].append(support_mask)

            query_word, query_pos1, query_pos2, query_mask = self.get_std_data(self.json_data[it]['meta_test'])
            query_set['word'].append(query_word)
            query_set['pos1'].append(query_pos1)
            query_set['pos2'].append(query_pos2)
            query_set['mask'].append(query_mask)
            query_label += [0] * Q

        # print(np.shape(support_set['pos1']), np.shape(support_set['pos1'][0]))
        support_set['word'] = np.stack(support_set['word'], 0)
        support_set['pos1'] = np.stack(support_set['pos1'], 0)
        support_set['pos2'] = np.stack(support_set['pos2'], 0)
        support_set['mask'] = np.stack(support_set['mask'], 0)
        query_set['word'] = np.concatenate(query_set['word'], 0)
        query_set['pos1'] = np.concatenate(query_set['pos1'], 0)
        query_set['pos2'] = np.concatenate(query_set['pos2'], 0)
        query_set['mask'] = np.concatenate(query_set['mask'], 0)
        query_label = np.array(query_label)

        return support_set, query_set, query_label

    def next_batch(self, B=4, it=0, N=5, K=1, Q=1):
        support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        label = []
        for one_sample in range(B):
            current_support, current_query, current_label = self.next_one(it, N=5, K=1, Q=1)
            support['word'].append(current_support['word'])
            support['pos1'].append(current_support['pos1'])
            support['pos2'].append(current_support['pos2'])
            support['mask'].append(current_support['mask'])
            query['word'].append(current_query['word'])
            query['pos1'].append(current_query['pos1'])
            query['pos2'].append(current_query['pos2'])
            query['mask'].append(current_query['mask'])
            label.append(current_label)

        support['word'] = torch.from_numpy(np.stack(support['word'], 0)).long().view(-1, self.max_length)
        support['pos1'] = torch.from_numpy(np.stack(support['pos1'], 0)).long().view(-1, self.max_length)
        support['pos2'] = torch.from_numpy(np.stack(support['pos2'], 0)).long().view(-1, self.max_length)
        support['mask'] = torch.from_numpy(np.stack(support['mask'], 0)).long().view(-1, self.max_length)
        query['word'] = torch.from_numpy(np.stack(query['word'], 0)).long().view(-1, self.max_length)
        query['pos1'] = torch.from_numpy(np.stack(query['pos1'], 0)).long().view(-1, self.max_length)
        query['pos2'] = torch.from_numpy(np.stack(query['pos2'], 0)).long().view(-1, self.max_length)
        query['mask'] = torch.from_numpy(np.stack(query['mask'], 0)).long().view(-1, self.max_length)
        label = torch.from_numpy(np.stack(label, 0).astype(np.int64)).long()

        for key in support:
            support[key] = support[key].cuda()
        for key in query:
            query[key] = query[key].cuda()
        label = label.cuda()

        return support, query, label
