Filtering out Outliers in Learning to Rank
===============================

This code is for the ICTIR 2022 full paper [Filtering out Outliers in Learning to Rank](https://doi.org/10.1145/3539813.3545127).

Abstract
---

Outlier data points are known to negatively affect the learning process of regression or classification models. Yet, their impact on the learning-to-rank scenario has not been thoroughly investigated so far. In this work, we propose SOUR, a learning-to-rank method that detects and removes outliers before building an effective ranking model. We limit our analysis to gradient-boosting decision trees, where SOUR searches for outlier instances that are incorrectly ranked in several iterations of the learning process. Extensive experiments show that removing a limited number of outlier data instances before re-training a new model provides statistically significant improvements and that SOUR outperforms state-of-the-art de-noising and outlier detection methods.

Implementation
---

**SOUR** (<ins>S</ins>urrender on <ins>Ou</ins>tliers and <ins>R</ins>ank) is a consistent outliers detector and removal algorithm built on top of [LightGBM](https://github.com/microsoft/LightGBM).
The code implements SOUR and the two variants presented in the article: *last*-SOUR and 𝑝-SOUR.

Usage
---

**SOUR** is accessible as a Python package. The following are the arguments of the ``train``method of the ``SOUR`` class:
  - ``params``: specify the [LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html).
  - ``outliers_type``: the type of outlier documents to remove. Accepts: ``"neg"``, ``"pos"``, and ``"all"`` (see definition in the article).
  - ``start``: iteration to start tracking the outlier documents.
  - ``end``: iteration to stop tracking the outlier documents.
  - ``p_sour=1``: for 𝑝-SOUR variant: threshold frequency to consider an outlier document as a frequent outlier (defined in ``[0, 1]``).
  - ``last_sour=False``: for *last*-SOUR variant: remove only outliers found in the last (``end``) training iteration.
  - ``cutoff=None``: starting rank to consider a document an outlier (see definition in the article). If ``None``, the ``"eval_at"`` [LightGBM parameter](https://lightgbm.readthedocs.io/en/latest/Parameters.html#eval_at) is used.
  - ``min_neg_rel=0``: minimal relevance label to consider a document as not relevant for the query (see definition in the article).
  - ``idx_to_removed=None``: if provided, the outlier detection is skipped, and a training without the documents in ``idx_to_removed=None`` is performed.

Note that the ``train`` method accepts the [LightGBM arguments](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html) as ``**kwargs``.

Citation
---

```
@inproceedings{DBLP:conf/ictir/MarcuzziL022,
  author       = {Federico Marcuzzi and
                  Claudio Lucchese and
                  Salvatore Orlando},
  editor       = {Fabio Crestani and
                  Gabriella Pasi and
                  {\'{E}}ric Gaussier},
  title        = {Filtering out Outliers in Learning to Rank},
  booktitle    = {{ICTIR} '22: The 2022 {ACM} {SIGIR} International Conference on the
                  Theory of Information Retrieval, Madrid, Spain, July 11 - 12, 2022},
  pages        = {214--222},
  publisher    = {{ACM}},
  year         = {2022},
  url          = {https://doi.org/10.1145/3539813.3545127},
  doi          = {10.1145/3539813.3545127},
  timestamp    = {Mon, 26 Jun 2023 20:49:11 +0200},
  biburl       = {https://dblp.org/rec/conf/ictir/MarcuzziL022.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
