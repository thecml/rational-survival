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

from survnam.run_nam import *

if __name__ == "__main__":
    seed_everything(seed)  # random seed

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers)

    # cpu or gpu to train the base
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("load data")

    train, (x_test, y_test) = create_test_train_fold(dataset=dataset,
                                                     id_fold=id_fold,
                                                     n_folds=n_folds,
                                                     n_splits=n_splits,
                                                     regression=not regression)
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    logging.info("begin training")

    # ========================= Train Random Survival Forests ==========================================================
    X, y = load_gbsg2()

    grade_str = X.loc[:, "tgrade"].astype(object).values[:, np.newaxis]
    grade_num = OrdinalEncoder(categories=[["I", "II", "III"]]).fit_transform(grade_str)

    X_no_grade = X.drop("tgrade", axis=1)
    Xt = OneHotEncoder().fit_transform(X_no_grade)
    Xt.loc[:, "tgrade"] = grade_num

    scaler = MinMaxScaler()
    Xt = pd.DataFrame(scaler.fit_transform(Xt), columns=Xt.columns)

    random_state = 20
    X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.25, random_state=random_state)
    # print(X_train.head(3))

    # ============ Find Max Distances ========
    d_max = max_euclidean_distance(X_train)  # max euclidean distance among all samples after normalization

    numeric_list = ['age', 'estrec', 'pnodes', 'progrec', 'tsize']
    max_distances = max_variable_distances(X_train, numeric_list)
    # d_list = [d_max, max_distances['age'], max_distances['estrec'],
    #           max_distances['pnodes'], max_distances['progrec'], max_distances['tsize']]
    # print(d_list)  # [2.2164908688956717, 1.0, 0.9265734265734266, 1.0, 1.0, 1.0000000000000002]

    rsf = RandomSurvivalForest(n_estimators=1000,
                               min_samples_split=10,
                               min_samples_leaf=15,
                               max_features="sqrt",
                               n_jobs=-1,
                               random_state=random_state)
    rsf.fit(X_train, y_train)

    # X_test_sorted = X_test.sort_values(by=["pnodes", "age"])
    # X_test_sel = pd.concat((X_test_sorted.head(3), X_test_sorted.tail(3)))
    # pd.Series(rsf.predict(X_test_sel))

    # surv = rsf.predict_cumulative_hazard_function(X_test_sel, return_array=True)

    # for i, s in enumerate(surv):
    #     plt.step(rsf.event_times_, s, where="post", label=str(i))

    # plt.ylabel("Cumulative hazard")
    # plt.xlabel("Time in days")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # =========================== Nelson-Aalon Estimator ===============================================================
    from lifelines.fitters.nelson_aalen_fitter import NelsonAalenFitter

    nelson = NelsonAalenFitter()
    nelson.fit(durations=pd.DataFrame(y_train).time, event_observed=pd.DataFrame(y_train).cens)

    x_nelson = nelson.cumulative_hazard_.index
    y_nelson = nelson.cumulative_hazard_.NA_estimate

    # plt.step(nelson.cumulative_hazard_.index,
    #          nelson.cumulative_hazard_.NA_estimate,
    #          where="post",
    #          label=str(i))

    # plt.ylabel("Cumulative hazard")
    # plt.xlabel("Time in days")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # =========================== Train Neural Additive Model ==========================================================
    test_scores = []
    while True:
        try:
            (x_train, y_train), (x_validate, y_validate) = next(train)
            model = train_model(x_train, y_train, x_validate, y_validate, device, rsf, max_distances, y_nelson)
            metric, total_score, _ = evaluate(model, test_loader, device)
            test_scores.append(total_score)
            logging.info(f"fold {len(test_scores)}/{n_splits} | test | {metric}={test_scores[-1]}")
        except StopIteration:
            break

        logging.info(f"mean test score={test_scores[-1]}")
        
    # Test model
    feature_outputs = []
    for i, (x, y) in enumerate(test_loader, start=1):
         x, y = x.to(device), y.to(device)
         _, feature_nn_outputs = model.forward(x)
         feature_outputs.append(feature_nn_outputs)
    print(feature_outputs)
    
    #
    # 2022-08-29 14:39:14,796 begin training
    #           age    estrec  horTh=yes  ...   progrec     tsize  tgrade
    # 292  0.728814  0.062063        1.0  ...  0.142857  0.102564     0.5
    # 46   0.745763  0.005245        0.0  ...  0.002521  0.145299     0.5
    # 447  0.525424  0.004371        0.0  ...  0.003361  0.273504     1.0
    # [3 rows x 8 columns]
    # train | loss = 77621.84145: 100%|██████████| 479/479 [2:13:39<00:00, 16.74s/it]
    # 2022-08-29 16:54:24,779 epoch 0 | train | 77621.84145000383
    # 2022-08-29 16:54:24,825 epoch 0 | validate | MAE=0.578447988607745
    # train | loss = 2057.93643:   3%|▎         | 16/479 [04:29<2:07:13, 16.49s/it]
