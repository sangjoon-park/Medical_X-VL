
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

import sys

sys.path.append('../../')

from eval_zero import evaluate, bootstrap
from zero_shot import make, make_true_labels, run_softmax_eval


# ----- DIRECTORIES ------ #
cxr_filepath: str = '/home/depecher/PycharmProjects/CheXzero/data/chexpert_test.h5'  # filepath of chest x-ray images (.h5)
cxr_true_labels_path: Optional[
    str] = '/home/depecher/PycharmProjects/CheXzero/data/groundtruth.csv'  # (optional for evaluation) if labels are provided, provide path
model_dir: str = '/COVID_8TB/sangjoon/chexzero_checkpoint/20230320_proposed_seed42/best_4/'  # where pretrained models are saved (.pt)
predictions_dir: Path = Path('/home/depecher/PycharmProjects/CheXzero/predictions')  # where to save predictions
cache_dir: str = predictions_dir / "cached"  # where to cache ensembled predictions

context_length: int = 120

# ------- LABELS ------  #
# Define labels to query each image | will return a prediction for each label
# cxr_labels: List[str] = ['Atelectasis', 'Cardiomegaly',
#                          'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
#                          'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
#                          'Pneumothorax', 'Support Devices']

cxr_labels: List[str] = ['Atelectasis', 'Cardiomegaly', 'Edema', 'Consolidation', 'Fracture', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax']

# ---- TEMPLATES ----- #
# Define set of templates | see Figure 1 for more details
cxr_pair_template: Tuple[str] = ('{}', 'no {}')

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
        cxr_filepath: str,
        cxr_labels: List[str],
        cxr_pair_template: Tuple[str],
        cache_dir: str = None,
        save_name: str = None,
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
            cxr_filepath=cxr_filepath,
        )

        # path to the cached prediction
        if cache_dir is not None:
            if save_name is not None:
                cache_path = Path(cache_dir) / f"{save_name}_{model_name}.npy"
            else:
                cache_path = Path(cache_dir) / f"{model_name}.npy"

        # # if prediction already cached, don't recompute prediction
        # if cache_dir is not None and os.path.exists(cache_path):
        #     print("Loading cached prediction for {}".format(model_name))
        #     y_pred = np.load(cache_path)
        # else:  # cached prediction not found, compute preds
        print("Inferring model {}".format(path))
        test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

        y_pred = run_softmax_eval(model, test_true, loader, cxr_labels, cxr_pair_template)
        if cache_dir is not None:
            Path(cache_dir).mkdir(exist_ok=True, parents=True)
            np.save(file=cache_path, arr=y_pred)
        predictions.append(y_pred)

    # compute average predictions
    y_pred_avg = np.mean(predictions, axis=0)

    return predictions, y_pred_avg


# %%
predictions, y_pred_avg = ensemble_models(
    model_paths=model_paths,
    cxr_filepath=cxr_filepath,
    cxr_labels=cxr_labels,
    cxr_pair_template=cxr_pair_template,
    cache_dir=cache_dir,
)
# %%
# save averaged preds
pred_name = "chexpert_preds.npy"  # add name of preds
predictions_dir = predictions_dir / pred_name
np.save(file=predictions_dir, arr=y_pred_avg)

cxr_labels: List[str] = ['Atelectasis', 'Cardiomegaly', 'Edema', 'Consolidation', 'Fracture', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax']

# make test_true
test_pred = y_pred_avg
test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

# evaluate model
cxr_results = evaluate(test_pred, test_true, cxr_labels)
Atelectasis_auc = cxr_results['Atelectasis_auc']
Cardiomegaly_auc = cxr_results['Cardiomegaly_auc']
Consolidation_auc = cxr_results['Consolidation_auc']
Edema_auc = cxr_results['Edema_auc']
Effusion_auc = cxr_results['Pleural Effusion_auc']
Fracture_auc = cxr_results['Fracture_auc']
Pneumonia_auc = cxr_results['Pneumonia_auc']
Pneumothorax_auc = cxr_results['Pneumothorax_auc']
Mean_auc = (Atelectasis_auc + Cardiomegaly_auc + Consolidation_auc + Edema_auc + Effusion_auc + Fracture_auc + Pneumonia_auc + Pneumothorax_auc) / 8.

# boostrap evaluations for 95% confidence intervals
bootstrap_results = bootstrap(test_pred, test_true, cxr_labels)
# %%
# display AUC with confidence intervals

Atelectasis_auc = bootstrap_results[1]['Atelectasis_auc']['mean']
Cardiomegaly_auc = bootstrap_results[1]['Cardiomegaly_auc']['mean']
Consolidation_auc = bootstrap_results[1]['Consolidation_auc']['mean']
Edema_auc = bootstrap_results[1]['Edema_auc']['mean']
Effusion_auc = bootstrap_results[1]['Pleural Effusion_auc']['mean']
Fracture_auc = bootstrap_results[1]['Fracture_auc']['mean']
Pneumonia_auc = bootstrap_results[1]['Pneumonia_auc']['mean']
Pneumothorax_auc = bootstrap_results[1]['Pneumothorax_auc']['mean']
Mean_auc = (Atelectasis_auc + Cardiomegaly_auc + Edema_auc + Consolidation_auc + Effusion_auc + Fracture_auc + Pneumonia_auc + Pneumothorax_auc) / 8.
print('Model name: {} / Mean AUC: {}'.format(model_dir, Mean_auc))

bootstrap_results[1]
# %%
