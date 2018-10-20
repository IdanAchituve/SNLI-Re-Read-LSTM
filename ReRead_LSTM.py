import numpy as np
import random
import json
import re
from time import gmtime, strftime
from prettytable import PrettyTable

# set seeds
random.seed(666)
np.random.seed(666)

import dynet as dy


# biLSTM transducer class for section a
class ReRead_LSTM(object):
    # constructor for creating parameter collection
    def __init__(self, vocab_size, num_labels, LSTM_params, embed_vec, P_rows, model, improvement):

        # LSTM_layers - [#layers in the premise BiLSTM, #layers in the hypothesis BiLSTM, dimension of parameters]
        embed_size = LSTM_params[2]
        self.params = {}
        self.max_seq_len = P_rows
        self.params_size = LSTM_params[2]

        # lookup:
        self.params["lookup"] = model.add_lookup_parameters((vocab_size, embed_size))
        self.params["lookup"].init_from_array(embed_vec)

        # premise bi-LSTM parameter collection:
        self.fw_premise_builder = dy.VanillaLSTMBuilder(LSTM_params[0], embed_size, LSTM_params[2] / 2, model)
        self.bw_premise_builder = dy.VanillaLSTMBuilder(LSTM_params[0], embed_size, LSTM_params[2] / 2, model)

        # hypothesis bi-LSTM parameter collection:
        self.fw_hypo_builder = dy.VanillaLSTMBuilder(LSTM_params[1], embed_size, LSTM_params[2], model)
        self.bw_hypo_builder = dy.VanillaLSTMBuilder(LSTM_params[1], embed_size, LSTM_params[2], model)

        # attend vector
        self.params["fw_A_t0"] = model.add_parameters((self.max_seq_len))
        self.params["bw_A_t0"] = model.add_parameters((self.max_seq_len))

        # reRead params:
        self.params["fw_Wp"] = model.add_parameters((LSTM_params[2], LSTM_params[2]))
        self.params["fw_Wm"] = model.add_parameters((LSTM_params[2], LSTM_params[2]))  # out layer parameter collection:
        self.params["fw_Wc"] = model.add_parameters((LSTM_params[2], LSTM_params[2]))
        self.params["fw_Walpha"] = model.add_parameters((LSTM_params[2]))  # out layer parameter collection:
        self.params["bw_Wp"] = model.add_parameters((LSTM_params[2], LSTM_params[2]))
        self.params["bw_Wm"] = model.add_parameters((LSTM_params[2], LSTM_params[2]))  # out layer parameter collection:
        self.params["bw_Wc"] = model.add_parameters((LSTM_params[2], LSTM_params[2]))
        self.params["bw_Walpha"] = model.add_parameters((LSTM_params[2]))  # out layer parameter collection:

        # out layer parameter collection:
        self.params["W"] = model.add_parameters((num_labels, LSTM_params[2] * 2))
        self.params["b"] = model.add_parameters((num_labels))

    # create expressions and return output
    def __call__(self, premise, hypothesis, word2int, improvement):

        lookup = self.params["lookup"]  # get lookup parameters
        premise_seq = [lookup[word2int.get(i)] for i in premise]  # get embeddings of each word
        prem_seq_len = len(premise_seq)

        # get initial state
        fw_lstm_prem = self.fw_premise_builder.initial_state()
        bw_lstm_prem = self.bw_premise_builder.initial_state()


        # get output vectors of all time steps for the premise bi-lstm
        fw_lstm_prem_output = fw_lstm_prem.transduce(premise_seq)
        bw_lstm_prem_output = bw_lstm_prem.transduce(reversed(premise_seq))
        # get P vector (matrix) of the premise BiLSTM
        bi_prem_output = [dy.concatenate([fw1, bw1]) for fw1, bw1 in
                          zip(fw_lstm_prem_output, reversed(bw_lstm_prem_output))]

        # Pad with zeros all entries until the max length
        for i in range(prem_seq_len, self.max_seq_len):
            zero_vec = dy.zeros(self.params_size)
            bi_prem_output.append(zero_vec)

        # get the list as numpy array (each element will be a column)
        # dims: params_size x paded_max_seq_len (e.g., 300x82)
        if improvement == "2" or improvement == "3":
            P_mat = []
            P_mat.append(dy.concatenate_cols(bi_prem_output))
            bi_prem_output.reverse()
            P_mat.append(dy.concatenate_cols(bi_prem_output))
        else:
            P_mat = dy.concatenate_cols(bi_prem_output)

        fw_rlstm_hypo_output, bw_rlstm_hypo_output = self.get_rlstm_output(hypothesis, word2int, P_mat, prem_seq_len, improvement)

        # concatenate the output of the fw and the bw from each timestep
        bi_hypo_output = [dy.concatenate([fw1, bw1]) for fw1, bw1 in
                          zip(fw_rlstm_hypo_output, reversed(bw_rlstm_hypo_output))]
        mean_vec = dy.average(bi_hypo_output)

        # calcualate probabilities for the labels
        W_out = dy.parameter(self.params["W"])
        b_out = dy.parameter(self.params["b"])
        net_output = dy.softmax(W_out * mean_vec + b_out)

        return net_output

    def get_rlstm_output(self, hypothesis, word2int, P_mat_in, prem_seq_len, improvement):

        lookup = self.params["lookup"]  # get lookup parameters
        hypo_seq = [lookup[word2int.get(i)] for i in hypothesis]  # get embeddings of each word

        # get initial state
        fw_s0 = self.fw_hypo_builder.initial_state()
        bw_s0 = self.bw_hypo_builder.initial_state()

        # will get the last state each time
        fw_s = fw_s0
        bw_s = bw_s0

        # get fw parameter expressions
        fw_At_prev = dy.parameter(self.params["fw_A_t0"])
        fw_Wp = dy.parameter(self.params["fw_Wp"])
        fw_Wm = dy.parameter(self.params["fw_Wm"])
        fw_Wc = dy.parameter(self.params["fw_Wc"])
        fw_Walpha = dy.parameter(self.params["fw_Walpha"])
        bw_At_prev = dy.parameter(self.params["bw_A_t0"])
        bw_Wp = dy.parameter(self.params["bw_Wp"])
        bw_Wm = dy.parameter(self.params["bw_Wm"])
        bw_Wc = dy.parameter(self.params["bw_Wc"])
        bw_Walpha = dy.parameter(self.params["bw_Walpha"])

        # create mask for the attend vector to take into account only the length of the current sequence
        if prem_seq_len < self.max_seq_len:
            mask = dy.concatenate([dy.ones(prem_seq_len), dy.zeros(self.max_seq_len - prem_seq_len)])
            # bw_mask = dy.concatenate([dy.zeros(self.max_seq_len-prem_seq_len), dy.ones(prem_seq_len)])
        else:
            mask = dy.ones(prem_seq_len)
            # bw_mask = dy.ones(prem_seq_len)

        # calculate forward & backward mask
        At_mask_fw = dy.cmult(fw_At_prev, mask)
        At_mask_bw = dy.cmult(bw_At_prev, mask)

        if improvement == "2" or improvement == "3":
            if prem_seq_len < self.max_seq_len:
                bw_mask = dy.concatenate([dy.zeros(self.max_seq_len - prem_seq_len), dy.ones(prem_seq_len)])
            else:
                bw_mask = dy.ones(prem_seq_len)
            At_mask_bw = dy.cmult(bw_At_prev, bw_mask)


        if improvement == "2" or improvement == "3":
            P_mat = P_mat_in[0]
        else:
            P_mat = P_mat_in

        idx = 0
        fw_output_vec = []
        # calculate the new output with the attention of the fw lstm
        for word in hypo_seq:
            fw_s = fw_s.add_input(word)  # add input to the network
            h_t = fw_s.h()[0]  # get the output vector of the current timestep

            # get the output gate value:
            Weights = self.fw_hypo_builder.get_parameter_expressions()
            Wox = dy.select_rows(Weights[0][0], range(self.params_size * 2, self.params_size * 3))
            Woh = dy.select_rows(Weights[0][1], range(self.params_size * 2, self.params_size * 3))
            bo = dy.select_rows(Weights[0][2], range(self.params_size * 2, self.params_size * 3))
            if idx == 0:
                out_gate = dy.logistic(Wox * word + bo)
            else:
                h_t_prev = fw_s.prev().h()[0]
                out_gate = dy.logistic(Wox * word + Woh * h_t_prev + bo)

            # matrix multiplication - [params_size x max_len_seq] x [max_len_seq x 1]
            # m dim: params_size x 1
            mt = P_mat * At_mask_fw

            # get the new out vector
            m_gated = dy.cmult(dy.tanh(mt), out_gate)
            h_t_new = h_t + m_gated
            fw_output_vec.append(h_t_new)

            # calculate alpha
            alpha = dy.colwise_add(fw_Wp * P_mat, fw_Wm * mt)
            if idx > 0:
                s_t_prev = fw_s.prev().s()[0]
                alpha = dy.colwise_add(alpha, fw_Wc * s_t_prev)

            if improvement == "1" or improvement == "3":
                alpha = dy.tanh(alpha)

            # compute the next At
            At_fw = dy.transpose(dy.transpose(fw_Walpha) * alpha)
            At_fw_exp = dy.exp(At_fw)
            At_fw_exp_mask = dy.cmult(At_fw_exp, mask)
            At_mask_fw = dy.cdiv(At_fw_exp_mask, dy.sum_elems(At_fw_exp_mask))
            idx += 1


        if improvement == "2" or improvement == "3":
            P_mat = P_mat_in[1]
        else:
            P_mat = P_mat_in


        idx = 0
        bw_output_vec = []
        # calculate the new output with the attention of the bw lstm
        for word in reversed(hypo_seq):
            bw_s = bw_s.add_input(word)  # add input to the network
            h_t = bw_s.h()[0]  # get the output vector of the current timestep

            # get the output gate value:
            Weights = self.bw_hypo_builder.get_parameter_expressions()
            Wox = dy.select_rows(Weights[0][0], range(self.params_size * 2, self.params_size * 3))
            Woh = dy.select_rows(Weights[0][1], range(self.params_size * 2, self.params_size * 3))
            bo = dy.select_rows(Weights[0][2], range(self.params_size * 2, self.params_size * 3))
            if idx == 0:
                out_gate = dy.logistic(Wox * word + bo)
            else:
                h_t_prev = bw_s.prev().h()[0]
                out_gate = dy.logistic(Wox * word + Woh * h_t_prev + bo)

            # matrix multiplication - [params_size x max_len_seq] x [max_len_seq x 1]
            # m dim: params_size x 1
            mt = P_mat * At_mask_bw

            # get the new out vector
            m_gated = dy.cmult(dy.tanh(mt), out_gate)
            h_t_new = h_t + m_gated
            bw_output_vec.append(h_t_new)

            # calculate alpha
            alpha = dy.colwise_add(bw_Wp * P_mat, bw_Wm * mt)
            if idx > 0:
                s_t_prev = bw_s.prev().s()[0]
                alpha = dy.colwise_add(alpha, bw_Wc * s_t_prev)

            if improvement == "1" or improvement == "3":
                alpha = dy.tanh(alpha)

            # compute the next At
            At_bw = dy.transpose(dy.transpose(bw_Walpha) * alpha)
            At_bw_exp = dy.exp(At_bw)
            At_bw_exp_mask = dy.cmult(At_bw_exp, mask)
            At_mask_bw = dy.cdiv(At_bw_exp_mask, dy.sum_elems(At_bw_exp_mask))
            idx += 1

        return fw_output_vec, bw_output_vec

    # return the loss and do regularization
    def create_network_return_loss(self, premise, hypothesis, word2int, label2int, label, improvement):
        out = self(premise, hypothesis, word2int, improvement)
        loss = -dy.log(dy.pick(out, label2int.get(label)))
        return loss, out

    # return the loss and prediction on dev/test set
    def create_network_return_best(self, premise, hypothesis, word2int, label2int, label, improvement):
        out = self(premise, hypothesis, word2int, improvement)
        loss = -dy.log(dy.pick(out, label2int.get(label)))
        return loss, out

    # compute the regularization term
    def get_regularization(self, batch_preds, word2int):
        reg = 0
        # compute regularization on the parameters
        for key, value in self.params.iteritems():
            if key != "b" and key != "lookup":
                expression = dy.parameter(value)
                reg += dy.sum_elems(dy.pow(expression, dy.scalarInput(2)))
            if key == "lookup":
                for example in batch_preds:
                    premise = example[0]
                    hypothesis = example[1]
                    premise_seq = [value[word2int.get(i)] for i in premise]
                    hypothesis_seq = [value[word2int.get(i)] for i in hypothesis]
                    for exp in premise_seq:
                        reg += dy.sum_elems(dy.pow(exp, dy.scalarInput(2)))
                    for exp in hypothesis_seq:
                        reg += dy.sum_elems(dy.pow(exp, dy.scalarInput(2)))

        # compute regularization on the bilstm terms
        for i in range(2):
            weights_prem_fw = self.fw_premise_builder.get_parameter_expressions()[0][i]
            weights_prem_bw = self.bw_premise_builder.get_parameter_expressions()[0][i]
            weights_hypo_fw = self.fw_hypo_builder.get_parameter_expressions()[0][i]
            weights_hypo_bw = self.bw_hypo_builder.get_parameter_expressions()[0][i]
            reg += dy.sum_elems(dy.pow(weights_prem_fw, dy.scalarInput(2)))
            reg += dy.sum_elems(dy.pow(weights_prem_bw, dy.scalarInput(2)))
            reg += dy.sum_elems(dy.pow(weights_hypo_fw, dy.scalarInput(2)))
            reg += dy.sum_elems(dy.pow(weights_hypo_bw, dy.scalarInput(2)))

        return reg


# run model on dev set
def test_on_dev_set_model(dev, batch_size, snli_classifier, word2int, label2int, improvement):
    good = bad = 0.0
    cum_loss = 0.0
    num_examples = len(dev)
    idx = 0
    batch_size = 100  # set to fixed number since it does not matter on performance in testing time

    for example in dev:
        if idx % batch_size == 0:
            dy.renew_cg()  # create new computation graph for each batch
            batch_preds = []  # batch predictions list
            batch_losses = []

        premise = example[0]
        hypothesis = example[1]
        label = example[2]

        loss, sentence_prdictions = snli_classifier.create_network_return_best(premise, hypothesis, word2int, label2int,
                                                                               label, improvement)
        batch_losses.append(loss)
        batch_preds.append([premise, hypothesis, label2int.get(label), sentence_prdictions])

        # calc batch loss and print examples to log
        if idx % batch_size == (batch_size - 1) or idx == (num_examples - 1):

            # after accumulating the loss from the batch run forward-backward
            batch_loss = dy.esum(batch_losses) / batch_size  # sum the loss of the batch
            cum_loss += batch_loss.value()  # this calls forward on each sequence in the batch through the whole net

            # calculate the accuracy on the batch
            for sen_to_print in batch_preds:
                label = sen_to_print[2]
                pred = sen_to_print[3]

                out_vec_vals = pred.npvalue()  # transform to numpy array
                pred_class = np.argmax(out_vec_vals)  # get max value
                if label == pred_class:
                    good += 1
                else:
                    bad += 1

        idx += 1

    returned_values = [str(cum_loss / num_examples), str(good / (good + bad))]
    return returned_values


# train model
def train_model(train, dev, test, epochs, batch_size, reg_lambda, trainer, snli_classifier, word2int,
                label2int, per_log, training_sample, sample_type, improvement):
    from time import gmtime, strftime
    curr_time = str(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    num_examples = len(train)
    print(curr_time + ": starting training")

    # print to screen and to log files
    t = PrettyTable(['time', '#epochs', 'train loss', 'train_acc', 'dev loss', 'dev_acc', 'test loss', 'test acc'])

    sentence_count = 0
    # training code: batched.
    for epoch in range(epochs):
        cum_loss = 0
        good = bad = 0.0
        idx = 0

        if sample_type == "sequential":
            train_sample = train[0:int(round(num_examples * training_sample))]
        else:
            # sample exmples from the train for the current epoch
            random.shuffle(train)
            train_sample = [train[i] for i in
                            sorted(random.sample(xrange(num_examples), int(round(num_examples * training_sample))))]

        for example in train_sample:
            if idx % batch_size == 0:
                dy.renew_cg()  # create new computation graph for each batch
                batch_preds = []  # batch predictions list
                batch_losses = []

            premise = example[0]
            hypothesis = example[1]
            label = example[2]
            loss, sentence_prdictions = snli_classifier.create_network_return_loss(premise, hypothesis, word2int,
                                                                                   label2int, label, improvement)
            batch_losses.append(loss)
            batch_preds.append([premise, hypothesis, label2int.get(label), sentence_prdictions])

            if (idx % batch_size == (batch_size - 1)) or idx == (num_examples - 1):
                # after accumulating the loss from the batch run forward-backward
                if reg_lambda > 0.0:
                    regularization = reg_lambda * snli_classifier.get_regularization(batch_preds,
                                                                                     word2int)  # compute the regularization
                    batch_loss = regularization + dy.esum(batch_losses) / batch_size  # sum the loss of the batch
                else:
                    batch_loss = dy.esum(batch_losses) / batch_size  # sum the loss of the batch

                cum_loss += batch_loss.value()  # this calls forward on each sequence in the batch through the whole net
                batch_loss.backward()  # calculate gradients
                trainer.update()  # update parameters

                # calculate the accuracy on the batch
                for sen_to_print in batch_preds:
                    label = sen_to_print[2]
                    pred = sen_to_print[3]

                    out_vec_vals = pred.npvalue()  # transform to numpy array
                    pred_class = np.argmax(out_vec_vals)  # get max value
                    if label == pred_class:
                        good += 1
                    else:
                        bad += 1
            idx += 1

        # get train/dev/test loss and accuracy
        train_loss = str(cum_loss / num_examples)  # train loss
        train_accuracy = str(good / (good + bad))

        dev_loss, dev_accuracy = test_on_dev_set_model(dev, batch_size, snli_classifier, word2int, label2int, improvement)
        test_loss, test_accuracy = test_model_on_blind_set(test, batch_size, snli_classifier, word2int, label2int, improvement)

        # print to screen and to log
        curr_time = str(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        t.add_row(
            [curr_time, str(epoch + 1), train_loss, train_accuracy, dev_loss, dev_accuracy, test_loss, test_accuracy])
        print str(t)
        per_log.write(str(t))


# Read datasets
def read_data(fname):
    data = []  # list of lists. each list is [[sentence1],[sentence2],label]
    max_len = 0
    unique_words = set()  # in order to get relevant words from the embedding file

    with open(fname, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            # get relevant data
            sentence1 = str(json_obj["sentence1"])
            sentence2 = str(json_obj["sentence2"])
            gold_label = str(json_obj["gold_label"])
            if gold_label != "-":
                # get all words/tokens in the sentences
                sentence1_list = [token.strip().lower() for token in re.split('(\W+)?', sentence1) if token.strip()]
                sentence2_list = [token.strip().lower() for token in re.split('(\W+)?', sentence2) if token.strip()]
                # create a record to add to the dataset
                record = [sentence1_list] + [sentence2_list] + [gold_label]
                data.append(record)
                # save unique words
                for word in sentence1_list:
                    unique_words.add(word)
                for word in sentence2_list:
                    unique_words.add(word)
                # save the max length sentence
                if len(sentence1_list) > max_len:
                    max_len = len(sentence1_list)

    return data, unique_words, max_len


# Glorot Initiallizer
def GlorotInit(vocab_length, emb_vec_len):
    import math
    # initialization according to Xavier suggestion
    epsilon = math.sqrt(6.0) / math.sqrt(vocab_length + emb_vec_len)  # uniform range
    vec = np.random.uniform(-epsilon, epsilon, emb_vec_len)
    return vec


# read training set and change it accordingly
def get_embeddings(embedding_file, unique_words, embedding_size):
    curr_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print(curr_time + ": uploading embeddings")

    # number of unique words + 2 for the <UNK> and <PAD>
    vocab_length = len(unique_words) + 2

    # read relevant embeddings
    embed_vec = []
    vocab = []
    # read embeddings
    f = open(embedding_file)
    for line in f:
        sr = line.split()  # split by spaces or tabs
        word = sr[0].lower()  # get the word
        # get embeddings of relevant words only
        if word in unique_words:
            embed_vec.append(np.array([float(j) for j in sr[1:]]))
            vocab.append(word)

    # create embedding for words not appeared in Glove
    emb_vec_len = embedding_size
    for word in unique_words:
        if word not in vocab:
            weights = GlorotInit(vocab_length, emb_vec_len)
            embed_vec.append(weights)
            vocab.append(word)

    # add unk embedding vector
    if "<UNK>" not in vocab:
        vocab.append("<UNK>")
        weights = GlorotInit(vocab_length, emb_vec_len)
        embed_vec.append(weights)

    # add pad embedding vector
    if "<PAD>" not in vocab:
        vocab.append("<PAD>")
        weights = GlorotInit(vocab_length, emb_vec_len)
        embed_vec.append(weights)

    return np.asarray(embed_vec), vocab


# entailment main function
def entailment(train_file, dev_file, test_file, embed_file, epochs, eps, reg_lambda, batch_size, per_log,
               LSTM_params, training_sample, sample_type, improvement):
    curr_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print(curr_time + ": starting process")

    # read train and dev data sets
    train, train_words, max_len_train = read_data(
        train_file)  # read train data to list. each list item is a sentence. each sentence is a tuple
    dev, dev_words, max_len_dev = read_data(
        dev_file)  # read train data to list. each list item is a sentence. each sentence is a tuple
    test, test_words, max_len_test = read_data(
        test_file)  # read train data to list. each list item is a sentence. each sentence is a tuple
    P_rows = max([max_len_train, max_len_dev, max_len_test])

    # unify all unique words to one set and delete independent sets
    all_words = train_words.union(dev_words).union(test_words)
    del train_words
    del dev_words
    del test_words

    # get embeddings
    embed_vec, vocab = get_embeddings(embed_file, all_words, LSTM_params[2])

    # define vocabulary and help structures
    word2int = {w: i for i, w in enumerate(vocab)}
    label2int = {l: i for i, l in enumerate(["entailment", "neutral", "contradiction"])}
    vocab_size = len(vocab)
    num_labels = 3

    # create a classifier
    m = dy.ParameterCollection()
    trainer = dy.AdadeltaTrainer(m, eps)  # define trainer
    snli_classifier = ReRead_LSTM(vocab_size, num_labels, LSTM_params, embed_vec, P_rows, m, improvement)  # create classifier
    train_model(train, dev, test, epochs, batch_size, reg_lambda, trainer, snli_classifier, word2int,
                label2int, per_log, training_sample, sample_type, improvement)


def test_model_on_blind_set(test, batch_size, snli_classifier, word2int, label2int, improvement):
    good = bad = 0.0
    cum_loss = 0.0
    num_examples = len(test)
    idx = 0
    batch_size = 100  # set to fixed number since it does not matter on performance in testing time

    for example in test:
        if idx % batch_size == 0:
            dy.renew_cg()  # create new computation graph for each batch
            batch_preds = []  # batch predictions list
            batch_losses = []

        premise = example[0]
        hypothesis = example[1]
        label = example[2]

        loss, sentence_prdictions = snli_classifier.create_network_return_best(premise, hypothesis, word2int, label2int,
                                                                               label, improvement)
        batch_losses.append(loss)
        batch_preds.append([premise, hypothesis, label2int.get(label), sentence_prdictions])

        # calc batch loss and print examples to log
        if idx % batch_size == (batch_size - 1) or idx == (num_examples - 1):
            # after accumulating the loss from the batch run forward-backward
            batch_loss = dy.esum(batch_losses) / batch_size  # sum the loss of the batch
            cum_loss += batch_loss.value()  # this calls forward on each sequence in the batch through the whole net

            # calculate the accuracy on the batch
            for sen_to_print in batch_preds:
                label = sen_to_print[2]
                pred = sen_to_print[3]

                out_vec_vals = pred.npvalue()  # transform to numpy array
                pred_class = np.argmax(out_vec_vals)  # get max value
                if label == pred_class:
                    good += 1
                else:
                    bad += 1

        idx += 1

    test_loss = str(cum_loss / num_examples)  # train loss
    test_accuracy = str(good / (good + bad))

    return [test_loss, test_accuracy]


if __name__ == '__main__':

    import sys

    train_file = sys.argv[1]
    dev_file = sys.argv[2]
    test_file = sys.argv[3]
    embed_file = sys.argv[4]
    log_path = sys.argv[5]
    epochs = int(sys.argv[6])
    lr = float(sys.argv[7])
    reg_lambda = float(sys.argv[8])
    batch_size = int(sys.argv[9])
    parameters_size = int(sys.argv[10])
    training_sample = float(sys.argv[11])
    sample_type = str(sys.argv[12])
    improvement = str(sys.argv[13])

    # default:
    premise_layers = 1
    hypothesis_layers = 1

    str_params = "epsilon: " + str(lr) + "\n" + "batch_size: " + str(batch_size) + "\n" + "reg_lambda: " + \
                 str(reg_lambda) + "\n" + "epochs: " + str(epochs) + "\n" + \
                 "premise_layers: " + str(premise_layers) + "\n" + "hypothesis_layers: " + str(hypothesis_layers) \
                 + "\n" + "parameters_size: " + str(parameters_size) + "\n" + "training sample: " \
                 + str(training_sample) + "\n" + "sample type: " + sample_type + "\n" + "improvement: " + improvement + "\n"

    LSTM_params = [premise_layers, hypothesis_layers, parameters_size]

    # initialize logs:
    curr_time = str(strftime("%Y-%m-%d_%H-%M-%S", gmtime()))

    # create log files
    per_log = open(log_path + curr_time, 'w')  # "./log/new_embed_performance_"
    per_log.write(str_params)

    entailment(train_file, dev_file, test_file, embed_file, epochs, lr, reg_lambda, batch_size,
               per_log, LSTM_params, training_sample, sample_type, improvement)