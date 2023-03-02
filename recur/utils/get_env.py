# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import csv
import json
import random
import argparse
from datetime import datetime

import numpy as np

import torch
import os
import pickle

import recur.src
from recur.src.slurm import init_signal_handler, init_distributed_mode
from recur.src.utils import bool_flag, initialize_exp
from recur.src.model import check_model_params, build_modules
# from recur.src.envs import ENVS, build_env
from recur.src.envs import ENVS, build_env
from recur.src.trainer import Trainer
from recur.src.evaluator import Evaluator
import recur.src.envs.generators as generators

np.seterr(all='raise')


# os.environ['CUDA_VISIBLE_DEVICES'] ='2'

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Recurrence prediction", add_help=False)

    # main parameters
    parser.add_argument("--dump_path", type=str, default=".",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="train",
                        help="Experiment name")
    parser.add_argument("--print_freq", type=int, default=10,
                        help="Print every n steps")
    parser.add_argument("--save_periodic", type=int, default=1,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="test_lr",
                        help="Experiment ID")

    # float16 / AMP API
    parser.add_argument("--fp16", type=bool_flag, default=True,
                        help="Run model with float16")
    parser.add_argument("--amp", type=int, default=1,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")
    # 使用 AMP 包装器进行 float16 / 分布式 / 梯度累积。 优化级别
    # model parameters
    parser.add_argument("--enc_emb_dim", type=int, default=128,
                        help="Encoder embedding layer size")
    parser.add_argument("--dec_emb_dim", type=int, default=128,
                        help="Decoder embedding layer size")
    parser.add_argument("--n_enc_layers", type=int, default=2,
                        help="Number of Transformer layers in the encoder")
    parser.add_argument("--n_dec_layers", type=int, default=2,
                        help="Number of Transformer layers in the decoder")
    parser.add_argument("--n_enc_heads", type=int, default=8,
                        help="Number of Transformer encoder heads")
    parser.add_argument("--n_dec_heads", type=int, default=8,
                        help="Number of Transformer decoder heads")
    parser.add_argument("--n_enc_hidden_layers", type=int, default=1,
                        help="Number of FFN layers in Transformer encoder")
    parser.add_argument("--n_dec_hidden_layers", type=int, default=1,
                        help="Number of FFN layers in Transformer decoder")

    parser.add_argument("--norm_attention", type=bool_flag, default=False,
                        help="Normalize attention and train temperaturee in Transformer")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")  # 使用正弦嵌入

    # training parameters

    parser.add_argument("--curriculum_n_ops", type=bool, default=True,
                        help="Whether we use a curriculum strategy for the number of ops during training")
    # 我们是否对训练期间的操作数量使用课程策略

    parser.add_argument("--env_base_seed", type=int, default=-1,
                        help="Base seed for environments (-1 to use timestamp seed)")
    parser.add_argument("--test_env_seed", type=int, default=1,
                        help="Test seed for environments")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Number of sentences per batch")
    parser.add_argument("--batch_size_eval", type=int, default=None,
                        help="Number of sentences per batch during evaluation (if None, set to 1.5*batch_size)")
    parser.add_argument("--optimizer", type=str, default="adam_inverse_sqrt,lr=0.00001,warmup_updates=10000",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=1,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=10000,
                        help="Epoch size / evaluation frequency")
    parser.add_argument("--max_epoch", type=int, default=1000,
                        help="Number of epochs")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")  # 验证指标
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")  # 在 N 次迭代中累积模型梯度
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of CPU workers for DataLoader")

    # export data / reload it
    parser.add_argument("--export_data", type=bool_flag, default=False,
                        help="Export data and disable training.")
    parser.add_argument("--reload_data", type=str, default="recurrence,/home/skm21/recur-main/data/test/10000.txt,/home/skm21/recur-main/data/valid/1000.txt,/home/skm21/recur-main/data/test/10000.txt",
                        help="Load dataset from the disk (task1,train_path1,valid_path1,test_path1;task2,train_path2,valid_path2,test_path2)")
    parser.add_argument("--reload_size", type=int, default="-1",
                        help="Reloaded training set size (-1 for everything)")
    parser.add_argument("--batch_load", type=bool_flag, default=False,
                        help="Load training set by batches (of size reload_size).")

    # environment parameters
    parser.add_argument("--env_name", type=str, default="recurrence",
                        help="Environment name")
    ENVS[parser.parse_known_args()[0].env_name].register_args(parser)

    # tasks
    parser.add_argument("--tasks", type=str, default="recurrence",
                        help="Tasks")

    # beam search configuration
    parser.add_argument("--beam_eval", type=bool_flag, default=True,
                        help="Evaluate with beam search decoding.")
    parser.add_argument("--max_output_len", type=int, default=64,
                        help="Max output length")
    parser.add_argument("--beam_eval_train", type=int, default=0,
                        help="At training time, number of validation equations to test the model on using beam search (-1 for everything, 0 to disable)")
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--beam_length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--beam_early_stopping", type=bool_flag, default=True,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # reload pretrained model / checkpoint
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")
    # evaluation
    parser.add_argument("--eval_size", type=int, default=1000,
                        help="Size of valid and test samples")
    parser.add_argument("--train_noise", type=float, default=0,
                        help="Amount of noise at train time")
    parser.add_argument("--eval_noise", type=float, default=0,
                        help="Amount of noise at test time")
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    parser.add_argument("--eval_from_exp", type=str, default="/home/skm21/recur-main/train/500M_固定训练数据_lr0.0005_3",
                        help="Path of experiment to use")
    parser.add_argument("--eval_data", type=str, default="/home/skm21/recur-main/data/valid/1000.txt",
                        help="Path of data to eval")
    parser.add_argument("--eval_verbose", type=int, default=0,
                        help="Export evaluation details")
    parser.add_argument("--eval_verbose_print", type=bool_flag, default=False,
                        help="Print evaluation details")
    parser.add_argument("--eval_input_length_modulo", type=int, default=-1,
                        help="Compute accuracy for all input lengths modulo X. -1 is equivalent to no ablation")

    # debug
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")
    parser.add_argument("--debug", help="Enable all debug flags",
                        action="store_true")

    # CPU / multi-gpu / multi-node
    parser.add_argument("--cpu", type=bool_flag, default=False,
                        help="Run on CPU")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")
    parser.add_argument("--nvidia_apex", type=bool_flag, default=False,
                        help="NVIDIA version of apex")

    parser.add_argument("--dl", type=str, default="0", help="指定gpu编号")

    return parser


def main(params):
    # initialize the multi-GPU / multi-node training
    # initialize experiment / SLURM signal handler for time limit / pre-emption
    init_distributed_mode(params)
    # logger = initialize_exp(params)
    # if params.is_slurm_job:
    #     init_signal_handler()

    # CPU / CUDA
    # if not params.cpu:
    #     assert torch.cuda.is_available()
    # src.utils.CUDA = not params.cpu

    # build environment / modules / trainer / evaluator
    if params.batch_size_eval is None: params.batch_size_eval = int(1.5 * params.batch_size)

    time_id = get_time_id()

    params.test_result_name = "result_" + time_id

    env = build_env(params)

    # modules = build_modules(env, params)
    # trainer = Trainer(modules, env, params)
    # evaluator = Evaluator(trainer)
    # envs=evaluator.ENV
    src = "+ 7 - 8 - 14 + 2 + 9 - 6 - 12 + 4 + 11 - 4 - 10 + 6 + 13 - 2 - 8 + 8 + 15 + 0 - 6 + 10 + 17 + 2 - 4 + 12 + 19 + 4 - 2 + 14 + 21"
    src = src.split(' ')
    tgt = "add sub relu n x_0_2 mul INT+ 1 0 INT- 1"
    tgt = tgt.split(" ")
    hyp = "add sub relu n x_0_2 mul INT+ 1 0 INT- 1"
    hyp = hyp.split(" ")
    tree = "((relu(n) sub x_0_2) add (10 mul -1))"
    error = env.check_prediction(src, tgt, hyp, tree, 10)
    print(error)
    exit()

    # training

def get_env():
    parser = get_parser()
    params = parser.parse_args()
    # init_distributed_mode(params)

    if params.batch_size_eval is None: params.batch_size_eval = int(1.5 * params.batch_size)

    time_id = get_time_id()

    params.test_result_name = "result_" + time_id

    env = build_env(params)
    return env

def txt_to_csv(path):
    with open(f"/home/skm21/recur-main/{params.exp_name}/{params.exp_id}/result.txt", 'r',
              encoding='utf-8-sig') as f:
        with open(f"/home/skm21/recur-main/{params.exp_name}/{params.exp_id}/result.csv", 'w',
                  encoding='utf-8-sig') as f2:
            writer = csv.writer(f2)
            lines = f.readlines()
            for line in lines:
                line = line.split()
                writer.writerow(line)


def obtain_parser():  # 获取参数
    parser = get_parser()
    params = parser.parse_args()
    return params


def obatain_data(params):
    # params = obtain_parser()

    init_distributed_mode(params)
    # logger = initialize_exp(params)
    if params.is_slurm_job:
        init_signal_handler()

    # CPU / CUDA
    if not params.cpu:
        assert torch.cuda.is_available()
    recur.src.utils.CUDA = not params.cpu

    # build environment / modules / trainer / evaluator
    if params.batch_size_eval is None: params.batch_size_eval = int(1.5 * params.batch_size)

    env = build_env(params)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)

    rng = np.random.RandomState(894965460)
    generator = generators.RandomRecurrence(params)
    tree, series, predictions, n_input_points = generator.generate(rng=rng, nb_ops=None, prediction_points=True,
                                                                   length=None)

    print(tree)
    print(series)
    print(predictions)
    print(n_input_points)
    rng = np.random.RandomState(89496543)
    tree, series, predictions, n_input_points = generator.generate(rng=rng, nb_ops=None, prediction_points=True,
                                                                   length=None)

    print(tree)
    print(series)
    print(predictions)
    print(n_input_points)


def get_time_id():
    time_now = datetime.now()
    time_id = str(time_now)[:19].replace(" ", '_')
    # time_id=time_id.replace(":","-")
    time_id = time_id[:11] + time_id[11:13] + 'h' + time_id[14:16] + 'min' + time_id[-2:] + 's'
    return time_id


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    main(params)
    exit()
    time_id = get_time_id()

    params.test_result_name = "result_" + time_id
    eval_path = params.eval_data
    name = params.test_result_name

    if params.eval_only and params.eval_from_exp != "":  # 使用保存下来的模型进行测试
        if os.path.exists(params.eval_from_exp + '/best-' + params.validation_metrics + '.pth'):
            params.reload_model = params.eval_from_exp + '/best-' + params.validation_metrics + '.pth'
        elif os.path.exists(params.eval_from_exp + "/checkpoint.pth"):
            params.reload_model = params.eval_from_exp + "/checkpoint.pth"
            # elif os.path.exists(params.eval_from_exp+"/periodic-24.pth"):
            #     params.reload_model = params.eval_from_exp+"/periodic-24.pth"
            print('#' * 50)
            print("params.reload_model ", params.reload_model)
            print(type(params.reload_model))
            print('#' * 50)
        else:
            raise

        eval_data = params.eval_data
        print("eval_data**", eval_data)
        # read params from pickle
        pickle_file = params.eval_from_exp + "/params.pkl"
        assert os.path.isfile(pickle_file)
        pk = pickle.load(open(pickle_file, 'rb'))
        pickled_args = pk.__dict__
        del pickled_args['exp_id']
        del pickled_args['dump_path']

        # for p in params.__dict__:#替换之前训练时使用的参数
        #     if p in pickled_args:
        #         params.__dict__[p] = pickled_args[p]

        print('#' * 50)
        print("params.reload_model ", params.reload_model)
        print(type(params.reload_model))
        print('#' * 50)

        eval_data = params.eval_data
        print("eval_data**", eval_data)

        params.eval_only = True
        params.eval_size = None

        if params.reload_data or params.eval_data:
            params.reload_data = params.tasks + ',' + eval_data + ',' + eval_data + ',' + eval_data
        params.is_slurm_job = False
        params.local_rank = -1
        params.master_port = -1
        params.num_workers = 1

    # debug mode
    if params.debug:
        params.exp_name = 'debug'
        if params.exp_id == '':
            params.exp_id = 'debug_%08i' % random.randint(0, 100000000)
        params.debug_slurm = True

    # check parameters
    check_model_params(params)

    # obatain_data(params) #获取数据，调试
    # exit()
    # run experiment
    main(params)


