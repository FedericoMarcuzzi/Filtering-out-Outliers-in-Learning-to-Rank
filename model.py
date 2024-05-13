import numpy as np
import lightgbm as lgb

from outliers_detector import OutliersFinder
from misc import prepare_lightgbm_sets, remove_docs, rename_dict_key

class SOUR():
    def __init__(self, queries, labels, qs_len, eval_set=None, eval_labels = None, eval_group=None, eval_names=None):
        self.queries = queries
        self.labels = labels
        self.qs_len = qs_len

        self.eval_set = eval_set
        self.eval_labels = eval_labels
        self.eval_group = eval_group
        self.eval_names = eval_names

    def train(self, params, outliers_type, start, end, p_sour=1, last_sour=False, cutoff=None, min_neg_rel=0, **kwargs):
        if cutoff is None:
            cutoff = params['eval_at']

        is_curr = True
        if type(end) is not list:
            end = [end]
            is_curr = False
        
        cleaned_params = rename_dict_key(params, "num_iterations", ["num_iterations", "num_iteration", "n_iter", "num_tree", "num_trees", "num_round", "num_rounds", "nrounds", "num_boost_round", "n_estimators", "max_iter"])
        cleaned_params = rename_dict_key(params, "early_stopping_round", ["early_stopping_rounds", "early_stopping", "n_iter_no_change"])

        outliers_finder = OutliersFinder(outliers_type, start, end, cutoff, min_neg_rel)
        train_set, valid_sets, valid_names = prepare_lightgbm_sets((self.queries, self.labels, self.qs_len), include_train=True)

        save_early_stopping_rounds = 0
        save_num_iterations = 100

        if "early_stopping_rounds" in cleaned_params:
            save_early_stopping_rounds = cleaned_params["early_stopping_rounds"]
        if "num_iterations" in cleaned_params:
            save_num_iterations = cleaned_params["num_iterations"]

        cleaned_params["early_stopping_rounds"] = 0
        cleaned_params["num_iterations"] = end[-1]
        
        model = lgb.train(cleaned_params, train_set, valid_sets=valid_sets, valid_names=valid_names, feval=outliers_finder)

        cleaned_params["early_stopping_rounds"] = save_early_stopping_rounds
        cleaned_params["num_iterations"] = save_num_iterations

        clean_queries, clean_labels, clean_qs_lens = remove_docs(self.queries, self.labels, self.qs_len, idx_to_removed)
        valid_sets, valid_names = prepare_lightgbm_sets((clean_queries, clean_labels, clean_qs_lens), [self.queries, self.labels, self.qs_len, ["train_set"]])
        
        model = None
        if "init_model" in kwargs:
            model = kwargs.pop("init_model")

        pre_end = 0
        for i, idx_to_removed in enumerate(outliers_finder.get_outliers_ids(self, p_sour=p_sour, last_sour=last_sour)):
            clean_queries, clean_labels, clean_qs_lens = remove_docs(self.queries, self.labels, self.qs_len, idx_to_removed)
            valid_sets, valid_names = prepare_lightgbm_sets((clean_queries, clean_labels, clean_qs_lens), [self.queries, self.eval_labels, self.eval_group, self.eval_names])        
            
            if is_curr:
                if i == len(end) - 1:
                    cleaned_params["num_iterations"] = save_num_iterations - pre_end if save_num_iterations > pre_end else save_num_iterations, 
                else:
                    cleaned_params["num_iterations"] = (end[i] - pre_end)
                    pre_end = end[i]

            model = lgb.train(cleaned_params, train_set, valid_sets=valid_sets, valid_names=valid_names, init_model=model, **kwargs)

        return model