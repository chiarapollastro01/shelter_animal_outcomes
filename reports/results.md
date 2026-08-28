# Results — Shelter Animal Outcomes

This report reads the artefacts of one full run of the pipeline. Reproduce it
with `snakemake --cores all`; the models, their sidecars and `metrics.json` are
regenerated on demand and are not committed.


### Experimental Setup & Run Parameters 

The grids explored are the
ones in `config.yaml` at the time of the run, and each model's sidecar records
the resolved parameters that produced it.

* **Data Splitting:**
  * **Final Test Split:** 20% hold-out from raw data (`test_size: 0.2`)
  * **Tournament Hold-out:** 20% validation split isolated per species before grid search (`holdout_size: 0.2`)
  * **Cross-Validation:** 5-fold Stratified CV (`cv_n_splits: 5`)
* **Optimization & Selection:**
  * **Tracked Metrics:** `f1_macro`, `balanced_accuracy`, `accuracy`
  * **Refit Target:** `f1_macro` (determines tournament winners)

### Hyperparameter Search Space

Each classifier pipeline combines the common preprocessing steps with model-specific grids:

* **Common Preprocessing Grid:**
  * Categorical rare-label threshold (`max_other_ratio`): `[0.10, 0.15, 0.20]`
  * SMOTE neighbours (`k_neighbors`): `[3, 5]`
* **Candidate Model Families:**
  * **Random Forest (`random_forest`):**
    * `n_estimators`: `[100, 200]`
    * `max_depth`: `[None, 15, 30]`
  * **K-Nearest Neighbours (`knn`):**
    * `n_neighbors`: `[3, 5, 11]`
    * `weights`: `["uniform", "distance"]`
  * **Logistic Regression (`logistic_regression`):**
    * `C`: `[0.1, 1.0, 10.0]`
    * `max_iter`: `[1000]`

---

## The winners

| | Dog | Cat |
| --- | --- | --- |
| Family | Random forest | Random forest |
| CV F1-macro | 0.4333 | 0.5177 |
| Hold-out F1-macro | 0.4270 | 0.5325 |
| `max_depth` | 15 | 15 |
| `n_estimators` | 100 | 100 |
| `max_other_ratio` | 0.15 | 0.15 |
| `smote__k_neighbors` | 3 | 3 |

The random forest wins both tournaments, ahead of the K-nearest-neighbours and
the logistic regression, indicating that the nonlinear interactions captured by the tree-based model are the most representative for this task. 

**`max_depth=15` beats `None` on both.** There was a possibility of having pure leaves (terminal node where 100% of the training samples belong to the exact same target class), but the model discarded it: unbounded depth could easily lead to memorizing of the training fold, which results in overfitting.


## The two scores agree

In order to avoid optimistic selection bias, where cross validation scores could be high due to favourable fold splits, a separate hold-out set was isolated prior to the grid search. 
The gap between cross-validated and hold-out score is 0.006 for dogs and
0.015 for cats. This number is fundamental to ensure a honest estimate, and this narrow discrepancy confirms that the tuning procedure did not overfit the validation folds. 


## Test-set metrics

| | Dog | Cat |
| --- | ---: | ---: |
| Accuracy | 0.5806 | 0.7728 |
| F1-weighted | 0.5817 | 0.7817 |
| F1-macro | 0.4084 | 0.5247 |
| Log loss | 0.9993 | 0.5921 |

### Accuracy is not the story

Predicting the majority class every time would score 0.417 on dogs and 0.494 on cats, as Adoption accounts for 41.7% of all dog outcomes while Transfer makes up 49.4% of cat cases. Against this baseline, the model gains ~16 percentage points on dogs and ~28 on cats, confirming meaningful learning on both tracks. Against that baseline the model earns 16 points on dogs and 28 on
cats, so it is learning something on both. But accuracy and F1-weighted both
reward the frequent classes, and the classes here are not close to balanced.

F1-macro is 17 points below accuracy on dogs and 25 below on cats, and that
distance is the whole reason F1-macro was chosen as the selection metric.
Optimising on accuracy would have rewarded a model that ignores the rare
outcomes altogether: `Died` holds 50 dog rows out of 15 595, and after the
species split, the hold-out and the cross-validation folds, a single fold sees
a handful of them. The macro average refuses to let those classes disappear
into the average.

### Per class

The previous metrics average over the five classes, whose support differs  
substantially. Therefore it's interestin to read one class at a time:

**Dogs** (3 119 test rows)

| Outcome | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Adoption | 0.68 | 0.69 | 0.69 | 1 300 |
| Transfer | 0.65 | 0.46 | 0.54 | 783 |
| Return_to_owner | 0.46 | 0.57 | 0.51 | 857 |
| Euthanasia | 0.28 | 0.31 | 0.30 | 169 |
| Died | 0.00 | 0.00 | 0.00 | 10 |

**Cats** (2 227 test rows)

| Outcome | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Adoption | 0.85 | 0.85 | 0.85 | 855 |
| Transfer | 0.86 | 0.81 | 0.83 | 1 101 |
| Euthanasia | 0.50 | 0.39 | 0.44 | 142 |
| Return_to_owner | 0.22 | 0.43 | 0.29 | 100 |
| Died | 0.22 | 0.21 | 0.21 | 29 |


**On dogs, `Died` is never predicted at all.** Precision, recall and F1 are all
zero on ten test rows. The class holds 50 rows in the whole dataset, and after
the species split, the hold-out and the cross-validation folds, a single fold
sees a handful: SMOTE interpolates between neighbours that are themselves too
few to describe the class. On cats the same outcome reaches 0.21 with 29 test
rows, the result is still poor, but no longer absent.

**The same class scores differently across species.** `Return_to_owner` is 0.51
on dogs and 0.29 on cats, and the difference is not in the outcome but in the
data: 857 test rows against 100. A cat that goes back to its owner is a rare
event, and the model treats it as one.

**Precision and recall lean in opposite directions on the two species.** On
dogs, `Transfer` has precision 0.65 and recall 0.46: the model is reluctant to
predict it and is usually right when it does. On cats, `Return_to_owner` has
precision 0.22 and recall 0.43: it is predicted twice as often as it should be,
which is the shape SMOTE's oversampling tends to produce on a minority class it
managed to learn something about.

This is what F1-macro was protecting against. Averaged with weights, the two
tables give 0.58 and 0.78 and look respectable; averaged evenly, they give 0.41
and 0.52, because a model that never predicts `Died` gets no credit for it.

### Cats are the easier problem

Metrics are better on cats, and the class distributions say why. Cats
concentrate in two outcomes, Transfer at 49% and Adoption at 38%. Dogs spread
across three, Adoption 42%, Return_to_owner 27% and Transfer 25%, so there is
simply more to get wrong. The exploratory phase had already shown this split
(`reports/eda.md`), which is why the two species get independent tournaments
rather than one model with a species feature.

### The log loss is worth reading

Log loss scores the predicted probabilities rather than the predicted labels, so it's an important metric for evaluation: it basically states
whether the model believed its own
answer. At 0.999 on dogs against 0.592 on cats, the dog model reflects ambiguity among the three dominant competing outcomes (Adoption, Transfer, and Return_to_owner).


