import numpy as np

class OutliersFinder():
    def __init__(self, outliers_type, start, end, cutoff, min_neg_rel=0):
        self.outliers_type = outliers_type
        self.start = start
        self.end = end
        self.cutoff = cutoff
        self.min_neg_rel = min_neg_rel # minimal doc label to consider a document as not-relevant

        self.all_outleirs_ids = [None] * len(end)
        self.last_outleirs_ids = None
        self.iteration = 0
        self.curr_end = 0

        if self.outliers_type not in ["neg", "pos", "all"]:
            raise "<outliers_type> must by 'neg', 'pos', or 'all'"

    def __call__(self, y_score, data):
        if self.curr_end < len(self.end):
            if self.iteration < self.end[self.curr_end]:
                if self.iteration >= self.start and data.params['name'] == 'train_set':
                    if self.all_outleirs_ids[self.curr_end] is None:
                        if self.curr_end == 0:
                            self.all_outleirs_ids[self.curr_end] = np.zeros(y_score.shape)
                        else:
                            self.all_outleirs_ids[self.curr_end] = np.copy(self.all_outleirs_ids[self.curr_end - 1])

                    self.last_outleirs_ids = self._compute_outliers_ids(data.label, y_score, data.group, self.cutoff)
                    self.all_outleirs_ids[self.curr_end][self.last_outleirs_ids] += 1
            else:
                self.curr_end += 1

        self.iteration += 1
        return '', 1, True

    # computes positive and negative outlier documents' IDs
    def _compute_outliers_ids(self, y_true, y_score, qs_len):
        n_docs = y_score.shape[0]
        idx_docs = np.arange(n_docs)
        cum = np.cumsum(qs_len)[:-1]

        ids_list = []
        for labels, query, idx in zip(np.array_split(y_true,cum), np.array_split(y_score,cum), np.array_split(idx_docs,cum)):
            size = len(query)
            cut = min(self.cutoff, size)
            rk = len(query) - 1 - np.argsort(query[::-1], kind='stable')[::-1] # stable argsort in descending order

            labels_sorted = labels[rk]
            idx_sorted = idx[rk]
            pos_idx = np.arange(size)
            
            # gets negative outlier documents' IDs
            if self.outliers_type != "pos":
                idx_stop = min(cut, np.where(labels_sorted > self.min_neg_rel)[0][-1]) if np.sum(labels_sorted > self.min_neg_rel) else -1
                neg_outliers_ids = idx_sorted[np.where((labels_sorted <= self.min_neg_rel) * (pos_idx < idx_stop))[0]]
                ids_list.append(neg_outliers_ids)

            # gets positive outlier documents' IDs
            if self.outliers_type != "neg":
                idx_max = max(cut, np.argmin(labels_sorted)) if np.sum(labels_sorted <= self.min_neg_rel) else len(labels_sorted)
                pos_outliers_ids = idx_sorted[np.where((labels_sorted > self.min_neg_rel) * (pos_idx >= idx_max))[0]]
                ids_list.append(pos_outliers_ids)

        return np.concatenate(ids_list)
    
    # gets consistent / frequent / last iteration outlier documents' IDs in the interval [start, end). NOTE: counting from zero.   
    def get_outliers_ids(self, p_sour=1, last_sour=False, curr_sour=False):
        if last_sour:
            return self.last_outleirs_ids
        
        if not curr_sour:
            return np.where(self.all_outleirs_ids[i] >= int(p_sour * (self.end[0] - self.start)))[0]
        
        empty_docs_idx = np.array([], dtype=int)
        idx_to_remove = np.zeros(self.all_outleirs_ids[0].shape)
        list_outleirs_ids = [np.where(self.all_outleirs_ids[i] >= int(p_sour * (end - self.start)))[0] for i, end in enumerate(self.end)]
        list_outleirs_ids.append(empty_docs_idx)

        for i, idx in enumerate(list_outleirs_ids):
            idx_to_remove[idx] += 1
            yield np.where(idx_to_remove > i)[0]
    
    def get_outliers_ids_from_score(self, y_true, y_score, qs_len):
        return self._compute_outliers_ids(self, y_true, y_score, qs_len)