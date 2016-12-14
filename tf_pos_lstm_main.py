from tf_pos_lstm_model import run_model, run_model_k_fold
import os
from argparse import ArgumentParser 
import pickle
import sys


parser = ArgumentParser()
subparsers = parser.add_subparsers(title='different modes', dest = 'mode', description='train or test')
train_parser=subparsers.add_parser('train', help='train parsing')
train_parser.add_argument("-lstm", "--lstm_srn", dest="lstm", help="rnn architecutre", type = int, default = 1)
train_parser.add_argument("-c", "--capitalize", dest="cap", help="head capitalization", type = int, default = 1)
train_parser.add_argument("-n", "--num_indicator", dest="num", help="number indicator", type = int, default = 1)
train_parser.add_argument("-b", "--bidirectional", dest="bi", help="bidirectional LSTM", type = int, default = 1)
train_parser.add_argument("-S", "--suffix", dest="suffix", help="suffix feature", type = int, default = 1)
train_parser.add_argument("-m", "--max_epochs",  dest="max_epochs", help="max_epochs", type=int, default = 100)
train_parser.add_argument("-N", "--num_layers",  dest="num_layers", help="number of layers", type=int, default = 2)
train_parser.add_argument("-cnn", "--cnn_layers",  dest="cnn_layers", help="number of layers for cnn", type=int, default = 2)
train_parser.add_argument("-u", "--units", dest="units", help="hidden units size", type=int, default = 64)
train_parser.add_argument("-e", "--embedding_name", dest="embedding_name", help="word embedding file name", default ="glovevector/glove.6B.100d.txt")
train_parser.add_argument("-d", "--embedding_dim", dest="embedding_dim", help="embedding dimension", type=int, default=100)
train_parser.add_argument("-s", "--seq_length", dest="seq_length", help="seq_length", type=int, default=-1)
#train_parser.add_argument("-l", "--expert_loss", dest="expert_loss", help="loss function ignore padding words", type= int, default = 0)
train_parser.add_argument("-E", "--seed", dest="seed", help="set seed", type= int, default = 0)
train_parser.add_argument("-L", "--longskip", dest="longskip", help="skip sentences of bigger size than the max", type=int, default = 1)
train_parser.add_argument("-t", "--trainable", dest="embedding_trainable", help="train embedding vectors", type=int, default = 1)
train_parser.add_argument("-j", "--jackknife_dim", dest="jackknife_dim", help="train embedding vectors", type=int, default = 5)
train_parser.add_argument("-J", "--jackknife", dest="jackknife", help="jackknife for supertagging?", type=int, default = 1)
train_parser.add_argument("-a", "--early_stopping", dest="early_stopping", help="early stopping", type=int, default = 2)
train_parser.add_argument("-A", "--attention", dest="attention", help="attention", type=int, default = 0)
train_parser.add_argument("-w", "--widow_size", dest="window_size", help="window size", type=int, default = 0)
train_parser.add_argument("-W", "--atwidow_size", dest="atwindow_size", help="attention window size", type=int, default = 5)
train_parser.add_argument("-M", "--suffix_dim", dest="suffix_dim", help="suffix_dim", type=int, default = 10)
train_parser.add_argument("-r", "--lrate", dest="lrate", help="lrate", type=float, default = 0.01)
train_parser.add_argument("-z", "--normalize", dest="normalize", help="normalize the loss based off of the sentence length", type = int, default = 0)
train_parser.add_argument("-R", "--recurrent_attention", dest="recurrent_attention", help="recurrent attention", type = int, default = 0)
train_parser.add_argument("-p", "--prob", dest="dropout_p", help="keep fraction", type=float, default = 1.0)
train_parser.add_argument("-sy", "--sync", dest="sync", help="synchronize input output dropout", type=int, default = 1)
train_parser.add_argument("-ep", "--embed_dropout", dest="embed_dropout", help="keep fraction of words", type=float, default = 1.0)
train_parser.add_argument("-hp", "--hidden_p", dest="hidden_p", help="keep fraction of hidden units", type=float, default = 1.0)
train_parser.add_argument("-i", "--input_prob", dest="input_dp", help="keep fraction for input", type=float, default = 1.0)
train_parser.add_argument("-T", "--Task", dest="task", help="supertagging or tagging", default='POS_models', choices=['POS_models', 'Super_models'])
test_parser=subparsers.add_parser('test', help='test parsing')
test_parser.add_argument("-d", "--model_dir", dest="model_dir", help="model directory")
test_parser.add_argument("-m", "--model_name", dest="modelname", help="model name")
test_parser.add_argument("-a", "--early_stopping", dest="early_stopping", help="early stopping", type=int, default = 2)
test_parser.add_argument("-n", "--non_training", dest="non_training", help= "non-training", type=int, default = 1)
test_parser.add_argument("-g", "--get_contexts", dest="get_contexts", help= "non-training", type=int, default = None)

opts = parser.parse_args()

if opts.mode == "train":
    
    embedding_type=os.path.dirname(opts.embedding_name) 
    if not embedding_type:
        embedding_type = opts.embedding_name

    if opts.task == 'Super_models':
        
        model_dir = 'Super_tagging_cap{0}_num{1}_bi{2}_numlayers{3}_embeddim{4}_embedtype{5}_seed{6}_units{7}_dropout{8}_inputdp{9}_embeddingtrain{10}_suffix{11}_windowsize{12}_jackknife{13}_jkdim{14}'.format(opts.cap, opts.num, opts.bi, opts.num_layers, opts.embedding_dim, embedding_type, opts.seed, opts.units, opts.dropout_p, opts.input_dp, opts.embedding_trainable, opts.suffix, opts.window_size, opts.jackknife, opts.jackknife_dim)
    else:
        opts.jackknife = 0 # no need for jackknife
        opts.jackknife_dim = 0 # no need for jackknife
        model_dir = 'POS_tagging_cap{0}_num{1}_bi{2}_numlayers{3}_embeddim{4}_embedtype{5}_seqlength{6}_seed{7}_longskip{8}_units{9}_lrate{10}_dropout{11}_inputdp{12}_embeddingtrain{13}_suffix{14}_windowsize{15}'.format(opts.cap, opts.num, opts.bi, opts.num_layers, opts.embedding_dim, embedding_type, opts.seq_length, opts.seed, opts.longskip, opts.units, opts.lrate, opts.dropout_p, opts.input_dp, opts.embedding_trainable, opts.suffix, opts.window_size)
    model_dir = os.path.join('..', opts.task, model_dir)
    if opts.attention >0:
        print('adding attention')
        model_dir += '_attention{0}'.format(opts.attention)
    if opts.attention == 3:
        print('attention window') 
        model_dir += '_atwindowsize{0}'.format(opts.atwindow_size)
    if opts.attention == 4:
        print('attention window cos difference') 
        model_dir += '_atwindowsize{0}'.format(opts.atwindow_size)
    if opts.attention == 5:
        print('attention window cos difference on both sides') 
        model_dir += '_atwindowsize{0}'.format(opts.atwindow_size)
    if opts.attention in [6, 7, 8, 9, 10, 11, 20, 21, 22, 40]:
        print('attention window cos difference on both sides with center vector') 
        model_dir += '_atwindowsize{0}'.format(opts.atwindow_size)
    if opts.attention == 30:
        print('convolutional attention')
        model_dir += '_cnnlayers{0}'.format(opts.cnn_layers)
        model_dir += '_atwindowsize{0}'.format(opts.atwindow_size)
    if opts.embed_dropout<1.0:
        print('add embed dropout')
        model_dir += '_embeddp{0}'.format(opts.embed_dropout)
    if opts.hidden_p < 1.0:
        model_dir += '_hiddendp{0}'.format(opts.hidden_p)
    if opts.sync == 1:
        model_dir += '_sync'
    if opts.lstm == 0:
        model_dir += '_SRN'
    if opts.lstm == 2:
        model_dir += '_GRU'
    if opts.recurrent_attention == 1:
        model_dir += '_rnnat'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    elif len(os.listdir(model_dir)) > 10:
        sys.exit('already trained')
    
    if opts.seq_length < 0:
        opts.seq_length = None
    if not opts.suffix:
        opts.suffix_dim = 0
    print(model_dir)
    setattr(opts, 'model_dir', model_dir) 
    options_file = os.path.join(model_dir, 'options.pkl')
    with open(options_file, 'wb') as foptions:
        pickle.dump(opts, foptions)
       # if statement here. OPTS = best model
    if opts.task == 'Super_models':
        if os.path.isfile('../k_fold_sec/predictions.pkl'):
            print('already done k_fold. Move on to jackknife training')
            run_model_k_fold(opts) # skip kfold 
        else:
            if not os.path.isdir('../k_fold_sec'):
                os.makedirs('../k_fold_sec')
            jk_opts_dir = '../POS_models/POS_tagging_cap1_num1_bi1_numlayers2_embeddim100_embedtypeglovevector_seqlength-1_seed0_longskip1_units128_lrate0.01_normalize0_dropout1.0_inputdp1.0_embeddingtrain1_suffix1_windowsize1/options.pkl'
            with open(jk_opts_dir, 'rb') as foptions:
                jk_opts=pickle.load(foptions)
            print('need to do k_fold')
            run_model_k_fold(opts, jk_opts) 
    else:
        print('just doing vanilla pos. no complication')
        run_model(opts)
        
    
    
    
if opts.mode == "test":
    modelname = os.path.join(opts.model_dir, opts.modelname)
    
    with open(os.path.join(opts.model_dir, 'options.pkl'), 'rb') as foptions:
        options=pickle.load(foptions)
    op_list = dir(options)
    # since we added new options, need to set them for old models
    if 'embed_dropout' not in op_list:
        options.embed_dropout = 1.0
    if 'hidden_p' not in op_list:   
        options.hidden_p = 1.0
    if 'sync' not in op_list:
        options.sync = 0
    if 'lstm' not in op_list:
        options.lstm = 1 
    
    if 'recurrent_attention' not in op_list:
        options.recurrent_attention = 0 
    if not opts==2 and not opts.non_training:
        print('change early stopping from {0} to {1}'.format(options.early_stopping, opts.early_stopping))
    
    options.early_stopping = opts.early_stopping
    data_so_far = opts.modelname.split('_')
    epoch = int(data_so_far[0][5:])
    best_accuracy = float(data_so_far[1][8:])
    if opts.non_training:
        options.max_epochs = epoch
    if options.task == 'Super_models':
        if opts.get_contexts is not None:
            opts.get_contexts = xrange(220) 
        run_model_k_fold(options, modelname = modelname, epoch = epoch, best_accuracy = best_accuracy, get_contexts = opts.get_contexts)
    else:
        run_model(options, modelname = modelname, epoch = epoch, best_accuracy = best_accuracy, saving_dir = '../k_fold_sec')
