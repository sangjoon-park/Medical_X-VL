
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

import sys

sys.path.append('../../')

from eval_zero import evaluate, bootstrap
from zero_error_mimic import make, make_true_labels, run_softmax_eval
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

def compute_cis(data, confidence_level=0.01):
    """
    FUNCTION: compute_cis
    ------------------------------------------------------
    Given a Pandas dataframe of (n, labels), return another
    Pandas dataframe that is (3, labels).

    Each row is lower bound, mean, upper bound of a confidence
    interval with `confidence`.

    Args:
        * data - Pandas Dataframe, of shape (num_bootstrap_samples, num_labels)
        * confidence_level (optional) - confidence level of interval

    Returns:
        * Pandas Dataframe, of shape (3, labels), representing mean, lower, upper
    """
    data_columns = list(data)
    intervals = []
    for i in data_columns:
        series = data[i]
        sorted_perfs = series.sort_values()
        lower_index = int(confidence_level / 2 * len(sorted_perfs)) - 1
        upper_index = int((1 - confidence_level / 2) * len(sorted_perfs)) - 1
        lower = sorted_perfs.iloc[lower_index].round(4)
        upper = sorted_perfs.iloc[upper_index].round(4)
        mean = round(sorted_perfs.mean(), 4)
        interval = pd.DataFrame({i: [mean, lower, upper]})
        intervals.append(interval)
    intervals_df = pd.concat(intervals, axis=1)
    intervals_df.index = ['mean', 'lower', 'upper']
    return intervals_df


# ----- DIRECTORIES ------ #
report_filepath: str = './data/openi/Test_selected.jsonl'

cxr_true_labels_path: Optional[
    str] = '/home/depecher/PycharmProjects/CheXzero/data/groundtruth.csv'  # (optional for evaluation) if labels are provided, provide path
model_dir: str = '/COVID_8TB/sangjoon/chexzero_checkpoint/20230320_exclude_testset_fuzz_42_sentencewise/best_5/'  # where pretrained models are saved (.pt)
predictions_dir: Path = Path('/home/depecher/PycharmProjects/CheXzero/predictions')  # where to save predictions
cache_dir: str = predictions_dir / "cached"  # where to cache ensembled predictions

context_length: int = 120

# ------- LABELS ------  #
# Define labels to query each image | will return a prediction for each label
# cxr_labels: List[str] = ['Atelectasis', 'Cardiomegaly',
#                          'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
#                          'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
#                          'Pneumothorax', 'Support Devices']

# cxr_labels: List[str] = ['Atelectasis', 'Cardiomegaly', 'Edema', 'Fracture', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax']

# ---- TEMPLATES ----- #
# Define set of templates | see Figure 1 for more details
# cxr_pair_template: Tuple[str] = ('{}', 'no {}')

# pos_query = ['Findings consistent with pneumonia', 'Findings suggesting pneumonia',
#              'This opacity can represent pneumonia', 'Findings are most compatible with pneumonia']
# neg_query = ['There is no pneumonia', 'No evidence of pneumonia',
#              'No evidence of acute pneumonia', 'No signs of pneumonia']

# ----- MODEL PATHS ------ #
# If using ensemble, collect all model paths
model_paths = []
for subdir, dirs, files in os.walk(model_dir):
    for file in files:
        full_dir = os.path.join(subdir, file)
        model_paths.append(full_dir)

print(model_paths)


# %% md
## Run Inference
# %%
## Run the model on the data set using ensembled models
def ensemble_models(
        model_paths: List[str],
        report_filepath: str,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Given a list of `model_paths`, ensemble model and return
    predictions. Caches predictions at `cache_dir` if location provided.

    Returns a list of each model's predictions and the averaged
    set of predictions.
    """

    predictions = []
    model_paths = sorted(model_paths)  # ensure consistency of
    for path in model_paths:  # for each model
        model_name = Path(path).stem

        # load in model and `torch.DataLoader`
        model, loader = make(
            model_path=path,
            report_filepath=report_filepath
        )

        # # path to the cached prediction
        # if cache_dir is not None:
        #     if save_name is not None:
        #         cache_path = Path(cache_dir) / f"{save_name}_{model_name}.npy"
        #     else:
        #         cache_path = Path(cache_dir) / f"{model_name}.npy"

        # # if prediction already cached, don't recompute prediction
        # if cache_dir is not None and os.path.exists(cache_path):
        #     print("Loading cached prediction for {}".format(model_name))
        #     y_pred = np.load(cache_path)
        # else:  # cached prediction not found, compute preds
        print("Inferring model {}".format(path))
        # test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

        y_pred, y_true = run_softmax_eval(model, loader)
        # if cache_dir is not None:
        #     Path(cache_dir).mkdir(exist_ok=True, parents=True)
        #     np.save(file=cache_path, arr=y_pred)
        predictions.append(y_pred)

    # compute average predictions
    y_pred_avg = np.mean(predictions, axis=0)

    return predictions, y_pred_avg, y_true


# %%
predictions, y_pred_avg, test_true = ensemble_models(
    model_paths=model_paths,
    report_filepath=report_filepath,
)
# %%
# save averaged preds
pred_name = "chexpert_preds.npy"  # add name of preds
predictions_dir = predictions_dir / pred_name
np.save(file=predictions_dir, arr=y_pred_avg)

# cxr_labels: List[str] = ['Atelectasis', 'Cardiomegaly', 'Edema', 'Fracture', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax']

# make test_true
test_pred = y_pred_avg
# test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

# evaluate model
# cxr_results = roc_auc_score()

auc = roc_auc_score(test_true, test_pred)
print(auc)

def bootstrap(y_pred, y_true, n_samples=1000, label_idx_map=None):
    '''
    This function will randomly sample with replacement
    from y_pred and y_true then evaluate `n` times
    and obtain AUROC scores for each.

    You can specify the number of samples that should be
    used with the `n_samples` parameter.

    Confidence intervals will be generated from each
    of the samples.

    Note:
    * n_total_labels >= n_cxr_labels
        `n_total_labels` is greater iff alternative labels are being tested
    '''
    np.random.seed(97)
    y_pred  # (500, n_total_labels)
    y_true  # (500, n_cxr_labels)

    idx = np.arange(len(y_true))

    boot_stats = []
    for i in range(n_samples):
        sample = resample(idx, replace=True, random_state=i)
        y_pred_sample = y_pred[sample]
        y_true_sample = y_true[sample]

        sample_stats = evaluate(y_pred_sample, y_true_sample)
        boot_stats.append(sample_stats)

    boot_stats = pd.concat(boot_stats)  # pandas array of evaluations for each sample
    return boot_stats, compute_cis(boot_stats)

# boostrap evaluations for 95% confidence intervals
bootstrap_results = bootstrap(test_pred, test_true)
# %%
# display AUC with confidence intervals

# Atelectasis_auc = bootstrap_results[1]['Atelectasis_auc']['mean']
# Cardiomegaly_auc = bootstrap_results[1]['Cardiomegaly_auc']['mean']
# # Consolidation_auc = bootstrap_results[1]['Consolidation_auc']['mean']
# Edema_auc = bootstrap_results[1]['Edema_auc']['mean']
# Effusion_auc = bootstrap_results[1]['Pleural Effusion_auc']['mean']
# Fracture_auc = bootstrap_results[1]['Fracture_auc']['mean']
# Pneumonia_auc = bootstrap_results[1]['Pneumonia_auc']['mean']
# Pneumothorax_auc = bootstrap_results[1]['Pneumothorax_auc']['mean']
# Mean_auc = (Atelectasis_auc + Cardiomegaly_auc + Edema_auc + Effusion_auc + Fracture_auc + Pneumonia_auc + Pneumothorax_auc) / 7.
# print('Model name: {} / Mean AUC: {}'.format(model_dir, Mean_auc))

bootstrap_results[1]
# %%
