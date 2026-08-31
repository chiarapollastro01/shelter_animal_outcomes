# Results — Shelter Animal Outcomes

This report reads the artefacts of one full run of the pipeline. Reproduce it with `snakemake --cores all`; the models, their sidecars and `metrics.json` are regenerated on demand and are not committed.
The tournament table below comes from the model sidecars in `models/`; the test-set figures come from `reports/metrics.json`, the aggregate scores from its `overall` block and the per-class tables from its `per_class` block.


## Experimental Setup & Run Parameters

The grids explored are the
ones in `config.yaml` at the time of the run, and each model's sidecar records the resolved parameters that produced it.

* **Data Splitting:**
  * **Final Test Split:** 20% hold-out from raw data (`test_size: 0.2`)
  * **Tournament Hold-out:** 20% validation split isolated per species before grid search (`holdout_size: 0.2`)
  * **Cross-Validation:** 5-fold Stratified CV (`cv_n_splits: 5`)
* **Optimization & Selection:**
  * **Tracked Metrics:** `f1_macro`, `balanced_accuracy`, `accuracy`
  * **Refit Target:** `f1_macro` (determines tournament winners)

## Hyperparameter Search Space

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
| CV F1-macro | 0.4330 | 0.5240 |
| Hold-out F1-macro | 0.4226 | 0.5380 |
| `max_depth` | 15 | 15 |
| `n_estimators` | 200 | 100 |
| `max_other_ratio` | 0.15 | 0.15 |
| `smote__k_neighbors` | 3 | 5 |

Two grid points can sit closer than the noise: an earlier run of this same code, on a different patch release of scikit-learn, crowned different but statistically indistinguishable hyperparameters. The exact versions behind these figures are frozen in `requirements-lock.txt`.

The CV F1-macro represents the mean score across the 5 validation folds used during hyperparameter tuning, while the hold-out score evaluates the winning configuration trained on the 80% split against the isolated 20% validation set. The final model artifact saved to disk refits this optimal configuration on 100% of the species dataset, as each sidecar records under `refit_on_full_species_data`.The test-set
tables below therefore come from a fully refitted model that no score in this intermediate table measures.

The random forest wins both tournaments, ahead of the K-nearest-neighbours and the logistic regression, indicating that the nonlinear interactions captured by the tree-based model are the most representative for this task. 

**`max_depth=15` beats `None` on both.** There was a possibility of having pure leaves (terminal node where 100% of the training samples belong to the exact same target class), but the model discarded it: unbounded depth could easily lead to memorizing of the training fold, which results in overfitting.


## The two scores agree

In order to avoid optimistic selection bias, where cross validation scores could be high due to favourable fold splits, a separate hold-out set was isolated prior to the grid search. 
The gap between cross-validated and hold-out score is 0.010 for dogs and
0.014 for cats. This number is fundamental to ensure an honest estimate, and this narrow discrepancy confirms that the tuning procedure did not overfit the validation folds. 

On cats the hold-out score is the higher of the two, which is the direction that costs nothing: the concern is a model that looks better in
cross-validation than it turns out to be, not the reverse.

## Test-set metrics

| | Dog | Cat |
| --- | ---: | ---: |
| Accuracy | 0.5816 | 0.7768 |
| Balanced accuracy | 0.4089 | 0.5403 |
| F1-weighted | 0.5822 | 0.7852 |
| F1-macro | 0.4095 | 0.5255 |
| Log loss | 0.9963 | 0.5802 |

### Accuracy is not the story

Predicting the majority class every time would score 0.417 on dogs and 0.494 on cats, as Adoption accounts for 41.7% of all dog outcomes while Transfer makes up 49.4% of cat cases. Against that baseline the model earns ~16 percentage points on dogs and ~28 on cats, so it is learning something on both. However, standard accuracy and weighted F1 are heavily driven by these high-frequency outcomes, masking the actual difficulty of the classification task under severe class imbalance.
This distortion becomes evident when evaluating unweighted metrics:
balanced accuracy, the mean of the per-class recalls, drops about 17 percentage points below standard accuracy for dogs and about 24 points for cats. The two numbers describe the same
predictions: the distance between them is what the frequent classes were
contributing. On dogs, balanced accuracy (0.4089) and F1-macro (0.4095) converge almost identically, as both refuse to scale a class's importance by its sample prevalence.
Optimising purely on accuracy would reward a model that ignores rare outcomes altogether: for example, Died accounts for only 50 dog records out of 15 595. After the species split, hold-out isolation, and 5-fold cross-validation, an individual fold sees barely a handful of these cases; macro-averaging prevents these critical minority classes from vanishing into the aggregate score.

Read against the right baselines, the macro score is less bleak than it looks. The 0.417 and 0.494 quoted above are *accuracies*; the same majority-class strategy scores far worse on the metric that actually selected the model:

| Strategy | Dog | Cat |
| --- | ---: | ---: |
| Always the majority class | 0.1177 | 0.1323 |
| Uniform random guess | 0.1632 | 0.1485 |
| Random, in proportion to the classes | 0.2000 | 0.2000 |
| **This model** | **0.4095** | **0.5255** |

Always predicting `Adoption` earns an F1 of 0.588 on that one class and zero on the other four, so the macro average lands at 0.118, which is the whole point of choosing F!-macro. The proportional baseline is 0.2 on both species and would be 0.2 on any dataset: guessing at each class's prevalence makes precision and recall both equal to it, so every F1 equals the prevalence and their mean is necessarily 1/K.

The ceiling matters too. `Died` contributes zero to the dog average and cannot realistically do otherwise on ten test rows, so four fifths is the most the macro score can reach: 0.4095 should be confronted with that.


### Per class

The previous metrics average over the five classes, whose support differs substantially. Therefore it is interesting to read one class at a time:

**Dogs** (3 119 test rows)

| Outcome | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Adoption | 0.69 | 0.69 | 0.69 | 1 300 |
| Transfer | 0.65 | 0.47 | 0.55 | 783 |
| Return_to_owner | 0.47 | 0.58 | 0.52 | 857 |
| Euthanasia | 0.29 | 0.30 | 0.29 | 169 |
| Died | 0.00 | 0.00 | 0.00 | 10 |

**Cats** (2 227 test rows)

| Outcome | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Adoption | 0.86 | 0.86 | 0.86 | 855 |
| Transfer | 0.86 | 0.81 | 0.83 | 1 101 |
| Return_to_owner | 0.23 | 0.44 | 0.30 | 100 |
| Euthanasia | 0.50 | 0.39 | 0.44 | 142 |
| Died | 0.19 | 0.21 | 0.20 | 29 |


**For dogs**, across the 10 test-set instances of Died, precision, recall, and F1 are all 0. The class holds 50 rows in the whole dataset, and after
the species split, the hold-out and the cross-validation folds, a single fold sees a handful: SMOTE interpolates between neighbours that are themselves too few to describe the class. For cats, the same class achieves an F1 of 0.20 across 29 test rows: still modest, but no longer completely unrepresented.

**Class predictability reflects real-world species dynamics.** `Return_to_owner` is 0.52 on dogs and 0.30 on cats, and the difference
stems directly from sample prevalence: 857 test rows against 100. A cat that goes back to its owner is a rare event, and the model treats it as one.

**Precision-recall trade-offs.**
While precision measures prediction purity (the proportion of predicted instances that are genuinely correct, penalising false alarms), recall measures detection coverage (the proportion of actual class instances successfully identified, penalising missed cases).
 On dogs, `Transfer` exhibits conservative behaviour with high precision (0.65) and lower recall (0.47): the model is reluctant to
predict it and is usually right when it does. On cats, `Return_to_owner` has low precision (0.23) and higher recall (0.44): it is predicted twice as often as it should be, it could be an artefact of SMOTE expanding synthetic decision boundaries around minority clusters.

This is what F1-macro was protecting against. Averaged with weights, the two tables give 0.58 and 0.79 and look respectable; averaged evenly, they give 0.41 and 0.53, because a model that never predicts `Died` gets no credit for it.

### Cats are the easier problem

Metrics are better on cats, and the class distributions say why. Cats
concentrate in two outcomes, Transfer at 49% and Adoption at 38%. Dogs spread across three, Adoption 42%, Return_to_owner 27% and Transfer 25%, so there is simply more to get wrong. The exploratory phase had already shown this split (`reports/eda.md`), which is why the two species get independent tournaments rather than one model with a species feature.

### The log loss is worth reading

Log loss scores the predicted probabilities rather than the predicted labels, so it's an important metric for evaluation: it basically states
whether the model believed its own
answer. At 0.996 on dogs against 0.580 on cats, the dog model reflects ambiguity among the three dominant competing outcomes (Adoption, Transfer, and Return_to_owner).


