# Dataset

## Provenance

Source: the [Shelter Animal Outcomes](https://www.kaggle.com/c/shelter-animal-outcomes)
competition on Kaggle, training split only (`train.csv`). The competition's
test split is not used here.

The records come from the Austin Animal Center and cover outcomes between
1 October 2013 and 21 February 2016. One row is one animal leaving the
shelter.

The file is not versioned with this repository: download it from the
competition page and place it at `data/raw_data/train.csv`. Its checksum is in
`checksums.txt` beside this file.

| | |
| --- | --- |
| Rows | 26 729 |
| Columns | 10 |
| Size | 2 824 793 bytes |

## Columns

| Column | Meaning |
| --- | --- |
| `AnimalID` | Unique ID assigned upon intake, carries no signal and it is dropped. |
| `Name` | Given name. Missing for animals that arrived unnamed. |
| `DateTime` | Timestamp of when the outcome was recorded, to the minute. |
| `OutcomeType` | **Target.** Five classes, see below. |
| `OutcomeSubtype` | Further detail on the outcome. Leaks the target and is dropped. |
| `AnimalType` | Species: `Dog` or `Cat`. Splits the analysis into two independent tracks. |
| `SexuponOutcome` | Sex and reproductive status in one string, e.g. `Neutered Male`. |
| `AgeuponOutcome` | Age of the animal at outcome time, free text, e.g. `2 years`, `3 weeks`. |
| `Breed` | Breed description, encodes crosses as `A/B` and mixed breed as `Mix`. |
| `Color` | Color description, encodes two-tone coats as `A/B`. |

## Target distribution

| Outcome | Cat | Dog | Total |

| Adoption | 4 272 | 6 497 | 10 769 |
| Transfer | 5 505 | 3 917 | 9 422 |
| Return_to_owner | 500 | 4 286 | 4 786 |
| Euthanasia | 710 | 845 | 1 555 |
| Died | 147 | 50 | 197 |

`Died` holds fewer than 200 rows in total, and only 50 of them are dogs.

## Missing values

| Column | Missing | Share |
| --- | ---: | ---: |
| `OutcomeSubtype` | 13 612 | 50.9% |
| `Name` | 7 691 | 28.8% |
| `AgeuponOutcome` | 18 | 0.1% |
| `SexuponOutcome` | 1 | <0.1% |
