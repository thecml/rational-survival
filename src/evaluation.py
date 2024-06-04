import sys,os
sys.path.append('./SurvivalEVAL/')
# sys.path.append('/home/weijiesun/survival_project/SurvivalEVAL/')
from Evaluator import LifelinesEvaluator

def ISD_evaluation(survival_curves, y_test, y_train, verbose=True):
    '''
    survival_curves: DataFrame shape[# timebin, # test]
    y_test: DataFrame[['time', 'event']]
    y_train: DataFrame[['time', 'event']]
    
    return: dict
    '''

    test_event_times = y_test["time"]
    test_event_indicators = y_test["event"]
    train_event_times = y_train["time"]
    train_event_indicators = y_train["event"]
    
    evaluator = LifelinesEvaluator(survival_curves, test_event_times, test_event_indicators,
                              train_event_times, train_event_indicators)

    ISD_eval_dict = {}
    cindex, _, _ = evaluator.concordance()

    mae_score = evaluator.mae(method="Pseudo_obs")

    mse_score = evaluator.mse(method="Hinge")

    # The largest event time is 52. So we use 53 time points (0, 1, ..., 52) to calculate the IBS
    ibs = evaluator.integrated_brier_score(num_points=survival_curves.shape[0], draw_figure=False)

    d_cal = evaluator.d_calibration()

    ISD_eval_dict['cindex'] = cindex
    ISD_eval_dict['mae_score'] = mae_score
    ISD_eval_dict['mse_score'] = mse_score
    ISD_eval_dict['ibs'] = ibs
    ISD_eval_dict['d_cal'] = d_cal
    
    if verbose:
        print ('cindex:', cindex)
        print ('mae_score:', mae_score)
        print ('mse_score:', mse_score)
        print ('ibs:', ibs)
        print ('d_cal:', d_cal)

    return ISD_eval_dict

