# MEDDOPLACE SCORER

This repository contains the official scorer for the [MEDDOPLACE Shared Task](https://temu.bsc.es/meddoplace).
MEDDOPLACE is a shared task/challenge and set of resources for the detection of locations, clinical departments, and related types of information such as nationalities or patient movements, in medical documents in Spanish.
For more information about the task, data, evaluation metrics, ... please visit the task's website.

## Requirements
To use this scorer, you'll need to have Python 3 installed in your computer. Clone this repository, create a new virtual environment and then install the required packages:

```bash
git clone https://github.com/darrylestrada97/meddoplace_scoring_script
cd meddoplace_scoring_script
python3 -m venv venv/
source venv/bin/activate
pip install -r requirements.txt
```

For Task 2.1 (Toponym Resolution using GeoNames), you will need the GeoNames *allCountries.txt* gazetteer that can be downloaded from [here](http://download.geonames.org/export/dump/allCountries.zip).

The MEDDOPLACE task data is available on [Zenodo](https://doi.org/10.5281/zenodo.7707566). Keep in mind that the reference test set won't be uploaded until the task has finished.

## Usage Instructions

This program compares two .TSV files, with one being the reference file (i.e. Gold Standard data provided by the task organizers)
and the other being the predictions or results file (i.e. the output of your system). Your .TSV file needs to have the same structure as the reference file,
which is explained in the [MEDDOPLACE Task Guide]().

For every submission you want to evaluate, you will need to create two folders: `ref/` and `res/`.
Place the reference file in the former and the predictions file in the latter, as in the example below:

```
meddoplace_submission/
+-- submission_1/
|   +-- ref/
|       +-- meddoplace_test_complete.tsv
|   +-- res/
|       +-- teambsc_submission_1.tsv
+-- submission_2/
|   +-- ref/
|       +-- meddoplace_test_complete.tsv
|   +-- res/
|       +-- teambsc_submission_2.tsv     
```

Once your file structure is ready, you can run the program like this:
```
python meddoplace_normalization.py -i ./folder_with_ref_res -o ./folder_to_output_scores -d <dir/to/allCountries.txt> -p GN
```

The output will be a scores.txt file, similar to the one outputted by CodaLab, with your evaluation results.

These are the possible arguments:

- `-i`: input folder where you've created your `ref/` and `res/` folders.
- `-o`: output folder where you want to save the scores.txt file.
- `-d`: path to the GeoNames gazetteer (only needed for task 2.1).
- `-p`: used for the subtask, it may be `GN`, `PC`, `SCTID`, `task1`, `task3` or `all`.
Use `GN` for task 2.1, `PC` for task 2.2, `SCTID` for task 2.3 and `all` for task 4.

For the normalization task, keep in mind that:
- The predictions file must include a column with the code source (GN, PC or SCTID) or else the program will fail.
- If any of your codes are not valid, the program will print a warning and count it as a False Negative.
- The scoring for task 2.1 will take longer than the others as the GeoNames gazetteer is quite big.

## Contact
If you have any questions or suggestions, please contact <salvador.limalopez [at] bsc.es>.
