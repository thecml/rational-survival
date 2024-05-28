import math
import os

import pandas as pd
import torch
import tqdm
import copy
import random
import logging
from absl import app
from absl import flags
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# import base.nam.metrics
# from base.nam import data_utils
from survnam import *
import numpy as np

FLAGS = flags.FLAGS

learning_rate = 1e-3  # "Hyper-parameter: learning rate."
decay_rate = 0.995  # "Hyper-parameter: Optimizer decay rate"
output_regularization = 0.0  # "Hyper-parameter: feature reg"
l2_regularization = 0.0  # "Hyper-parameter: l2 weight decay"
dropout = 0.5  # "Hyper-parameter: Dropout rate"
feature_dropout = 0.0  # "Hyper-parameter: Prob. with which features are dropped"

training_epochs = 1  # "The number of epochs to run training for." # def. 10
early_stopping_epochs = 5  # "Early stopping epochs" # def.  60
batch_size = 32  # "Hyper-parameter: batch size." # def. 1
data_split = 1  # "Dataset split index to use. Possible values are 1 to `num_splits`."
seed = 1  # "Seed used for reproducibility."
n_basis_functions = 1000  # "Number of basis functions to use in a FeatureNN for a real-valued feature."
units_multiplier = 2  # "Number of basis functions for a categorical feature"
n_models = 1  # "the number of models to train."
n_splits = 3  # "Number of data splits to use"
id_fold = 1  # "Index of the fold to be used"

hidden_units = []  # "Amounts of neurons for additional hidden layers, e.g. 64,32,32"
log_file = None  # "File where to store summaries."
dataset = "gbsg2"  # "Name of the dataset to load for training."
shallow_layer = "exu"  # "Activation function used for the first layer: (1) relu, (2) exu"
hidden_layer = "relu"  # "Activation function used for the hidden layers: (1) relu, (2) exu"
regression = True  # "Boolean for regression or classification"

n_folds = 5
N_GEN = 100


def generate_normal(mean, std, N=N_GEN):
    """

    :param mean:
    :param std:
    :param N:
    :return:
    """
    s = np.random.normal(mean, std, N)
    return s


def seed_everything(seed):
    """

    :param seed:
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(x_train, y_train, x_valid, y_valid, device, rsf, max_distances, nelson_est):
    """

    :param x_train:
    :param y_train:
    :param x_valid:
    :param y_valid:
    :param device:
    :return:
    """
    model = NeuralAdditiveModel(
        input_size=x_train.shape[-1],
        shallow_units=calculate_n_units(x_train, n_basis_functions, units_multiplier),
        hidden_units=list(map(int, hidden_units)),
        shallow_layer=ExULayer if shallow_layer == "exu" else ReLULayer,
        hidden_layer=ExULayer if hidden_layer == "exu" else ReLULayer,
        hidden_dropout=dropout,
        feature_dropout=feature_dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)
    # criterion = metrics.penalized_mse if regression else metrics.penalized_cross_entropy
    criterion = survnam_loss
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataset = TensorDataset(torch.tensor(x_valid), torch.tensor(y_valid))
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

    n_tries = early_stopping_epochs  # to restrict the minimum training epochs
    best_validation_score, best_weights = 0, None  # to store the optimal performance

    for epoch in range(training_epochs):
        model = model.train()  # training the base
        total_loss = train_one_epoch(model, criterion, optimizer, train_loader, device, rsf, max_distances, nelson_est)
        # record the log of training (training loss)
        logging.info(f"epoch {epoch} | train | {total_loss}")

        scheduler.step()  # update the learning rate

        model = model.eval()  # validating the base
        metric, val_score, _ = evaluate(model, validate_loader, device)
        # record the log of validation (validation score)
        logging.info(f"epoch {epoch} | validate | {metric}={val_score}")

        # early stopping if the validation performance degrades
        # but also restricted to a minimum epochs of training
        if val_score <= best_validation_score and n_tries > 0:
            n_tries -= 1
            continue
        elif val_score <= best_validation_score:
            logging.info(f"early stopping at epoch {epoch}")
            break

        best_validation_score = val_score  # update the optimal validation score
        best_weights = copy.deepcopy(model.state_dict())  # update the optimal base

    model.load_state_dict(best_weights)  # continue training from the optimal base

    return model


def train_one_epoch(model, criterion, optimizer, data_loader, device, rsf, max_distances, nelson_est):
    """

    :param model:
    :param criterion:
    :param optimizer:
    :param data_loader:
    :param device:
    :return:
    """

    # tqdm is a library in Python which is used for creating Progress Meters or Progress Bars.tqdm got its name from
    # the Arabic name taqaddum which means 'progress'.
    pbar = tqdm.tqdm(enumerate(data_loader, start=1), total=len(data_loader))

    # find all estimated CHF by nelson-aalon estimator corresponding to time points with events
    list_nelson = []
    for _, event_time in enumerate(rsf.unique_times_):
        # print(nelson_est.loc[72])  # 0.001988071570576899
        list_nelson.append(nelson_est.loc[event_time])
    # append the CHF by nelson-aalon estimator
    est_chf = pd.concat([pd.DataFrame(rsf.unique_times_), pd.DataFrame(list_nelson)], axis=1)

    # formulate the column names
    column_list = ['event_times', 'chf_nelson']
    for num in range(100):
        column_list.append('chf_rsf_{}'.format(num))

    total_loss = 0
    for i, (x, y) in pbar:
        x_loss = 0  # print("x", x[0][0].item()) # print("x", x)

        # ============================== Generate Points Following Normal Distribution =================================
        gen_age = generate_normal(x[0][0].item(), max_distances['age'] * 0.1)
        gen_estrec = generate_normal(x[0][1].item(), max_distances['estrec'] * 0.1)
        gen_pnodes = generate_normal(x[0][4].item(), max_distances['pnodes'] * 0.1)
        gen_progrec = generate_normal(x[0][5].item(), max_distances['progrec'] * 0.1)
        gen_tsize = generate_normal(x[0][6].item(), max_distances['tsize'] * 0.1)

        df_input = pd.concat([pd.DataFrame(gen_age), pd.DataFrame(gen_estrec)], axis=1)
        df_input['horTh=yes'] = x[0][2].item()
        df_input['menostat=Post'] = x[0][3].item()
        df_input = pd.concat([df_input, pd.DataFrame(gen_pnodes), pd.DataFrame(gen_progrec),
                              pd.DataFrame(gen_tsize)], axis=1)
        df_input['tgrade'] = x[0][7].item()
        df_input.columns = ['age', 'estrec', 'horTh=yes', 'menostat=Post', 'pnodes', 'progrec', 'tsize', 'tgrade']
        # print(df_input)  generated N (=100) points for the explained point

        # largest distance among all generated points
        d_max = max_euclidean_distance(df_input)

        # input = df_input.iloc[0].astype(np.float32)
        # input = torch.tensor(input.values)
        # # print("input", input)
        # x = torch.unsqueeze(input, dim=0)
        # # print("x", x)
        # x, y = x.to(device), y.to(device)
        # # print("x", x)  # normalized input features of one instance

        # logits, fnns_out = model.forward(x)
        # print("logits", logits)  # final additive outputs
        # print("fnns_out", fnns_out)  # outputs from each shape functions
        # print("logits", logits)  # logits tensor([1.1540], grad_fn=<AddBackward0>)

        # print("nelson_est", nelson_est, len(nelson_est), type(nelson_est))  # length = 454
        # 18.0      0.000000
        #             ...
        # 2471.0    1.028009
        # 2551.0    1.028009

        # print("rsf.event_times_", rsf.event_times_, len(rsf.event_times_), type(rsf.event_times_))  # length = 215
        # [72.   98.  120.  160.  169.  171.  173.  177.  180.  184.  191.  195.
        #  205.  223.  227.  233.  238.  241.  242.  247.  249.  272.  281.  286.
        #  288.  293.  307.  308.  329.  336.  338.  344.  348.  350.  357.  359.
        #  360.  369.  370.  372.  374.  375.  385.  392.  394.  403.  410.  415. ...

        # print("chf_rsf[0]", chf_rsf[0], len(chf_rsf[0]), type(chf_rsf[0]))  # length = 215
        # [0.00442582 0.00446031 0.00449479 0.00630778 0.00834145 0.00841288
        #  0.00850506 0.00923685 0.01042667 0.01404961 0.02457283 0.02506429
        #  0.02506429 0.0356179  0.03565238 0.03629387 0.03876501 0.03909379 ...

        # CHF estimated by RSF for generated points (N = 100)
        chf_rsf = rsf.predict_cumulative_hazard_function(df_input, return_array=True)
        # print(chf_rsf[0])  CHF of 0th instance by RSF
        # [0.00207913 0.00207913 0.00229458 0.00318554 0.00665146 0.00665146
        #  0.01027261 0.01223869 0.01479719 0.01945229 0.02890883 0.0289664
        #  0.03129134 0.03592671 0.037428   0.0419167  0.04567486 0.04567486 ...
        # print("chf_rsf", pd.DataFrame(chf_rsf).T)

        # chf = pd.DataFrame()
        # chf = pd.concat([chf, est_chf], axis=1)
        # append all CHF by random survival forest for generated points
        # for _, chf_generated_point in enumerate(chf_rsf):
        #     chf = pd.concat([chf, pd.DataFrame(chf_generated_point)], axis=1)
        # plt.step(rsf.event_times_, s, where="post", label=str(i))

        chf = pd.concat([est_chf, pd.DataFrame(chf_rsf).T], axis=1)
        chf.columns = column_list
        # print(chf.head(3))

        df_input.loc[N_GEN] = x[0].tolist()  # append the explained point to the end

        total_duration = chf['event_times'].loc[214]

        for k in range(N_GEN):

            xk = df_input.iloc[k].astype(np.float32)
            xk = torch.unsqueeze(torch.tensor(xk.values), dim=0)
            # x, y = x.to(device), y.to(device)
            xk = xk.to(device)
            logits, _ = model.forward(xk)

            # dx = sum((df_input.iloc[k] - df_input.iloc[N_GEN]) ** 2) ** 0.5
            weight_k = 1 - ((sum((df_input.iloc[k] - df_input.iloc[N_GEN]) ** 2) ** 0.5) / d_max) ** 0.5
            # print("weight_k", weight_k)  # weight_k 0.6658194611105693

            for j in range(215):
                if j == 0:
                    duration_j = chf['event_times'].loc[j] / total_duration
                    phi = 0
                else:
                    duration_j = (chf['event_times'].loc[j] - chf['event_times'].loc[j - 1]) \
                                 / total_duration
                    phi = math.log1p(chf['chf_rsf_{}'.format(k)].loc[j]) - math.log1p(chf['chf_nelson'].loc[j])

                # print("duration_j", duration_j)  # duration_j 0.0008431703204047217
                # print("phi", phi)  # phi 0.27634822347966037

                logits_j = logits * ((weight_k * duration_j) ** 0.5)
                # truths = phi * ((weight_k * duration_j) ** 0.5)
                truths = torch.as_tensor(phi * ((weight_k * duration_j) ** 0.5))
                # print("logits_j", logits_j)
                # print("truths", truths)

                loss = criterion(logits_j, truths)
                loss.backward(retain_graph=True)
                x_loss += loss.item()

            optimizer.step()
            model.zero_grad()

        # print("x_loss:", x_loss)
        # loss = criterion(logits, est_chf)

        # total_loss -= (total_loss / i) - (loss.item() / i)
        total_loss += x_loss

        pbar.set_description(f"train | loss = {total_loss:.5f}")

    return total_loss


def evaluate(model, data_loader, device):
    """

    :param model:
    :param data_loader:
    :param device:
    :return:
    """
    total_score = 0
    metric = None
    for i, (x, y) in enumerate(data_loader, start=1):
        x, y = x.to(device), y.to(device)
        logits, fnns_out = model.forward(x)
        metric, score = calculate_metric(logits, y, regression=regression)
        total_score -= (total_score / i) - (score / i)

    return metric, total_score, logits


def max_euclidean_distance(df):
    """
    Find the maximum Euclidean distance among instances in dataframe
    :param df: input dataframe with all instance
    :return: max distance among samples
    """
    max_distance = 0
    for former in range(df.shape[0]):
        for latter in range(former + 1, df.shape[0]):
            temp_distance = sum((df.iloc[latter] - df.iloc[former]) ** 2) ** 0.5
            max_distance = temp_distance if (temp_distance >= max_distance) else max_distance
    return max_distance


def max_variable_distances(df, variable_list):
    """
    Find the maximum absolute distances among selected variables
    :param df: input dataframe with all instances
    :param variable_list: selected variables for finding distances
    :return: max distances of variables
    """
    max_distance = {}
    for _, variable in enumerate(variable_list):
        max_distance[variable] = 0

    for former in range(df.shape[0]):
        for latter in range(former + 1, df.shape[0]):
            temp_distance = abs(df.iloc[latter] - df.iloc[former])
            for _, variable in enumerate(variable_list):
                max_distance[variable] = temp_distance[variable] if (temp_distance[variable] > max_distance[variable]) \
                    else max_distance[variable]
    return max_distance