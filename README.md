# Mercedes 
Kaggle Mercedes competition

PLEASE USE PYTHON 3.X


## Working setup
All scripts should run from the top level of this project.

###  Validation 
Due to the small size of the training data, validation is computed on a 20-fold cv for STD and MEAN scores
 

The `create_validation.py` is the driving split for the group to add code to the src directory.
Improvements in local scores on validation set give a concrete way accepting new code into the `src` dir
as such it is the most important part of the project, lb and local cv matching

### Modules used

* pandas
* numpy
* scikit-learn 
* keras
* xgboost
* lightgbm


### data
### data/raw
The raw unprocessed data, the combination of scripts from `src/dataset/build_DatasetV#.py` and 
`src/model/build_modelV#.py` will yeild the final repeatable solution

### data/processed
Location of processed data to be used moving forward, this could include the output of build_datasetV#` or 
metalevel stacking / ensemble base data, embeddings or pre-trained models etc.

### data/output
The final output to use for submissions


### data/tmp
A temp location to dump information, this area should not be used for any type of long term storage, more 
as a location to 'dump' temporary data that should be removed after the execution of a script.


### scratch
The scratch folder is where all work in progress by each individual should reside, this is part o teh .gitignore
and allows for people to investigate without polluting the repo.

### notebooks
The notebooks folder is for showing work to the wider group, it is not part of the final solution run, however it may 
can be used to show teh other members interesting findings from scratch work

### src
#### src/datasets
A series of `build_datasetV#.py` files which construct datasets to use for modelling from the raw data, 
these data sets should only include scripts which make the final ensemble - investigation should be completed in 
scratch
#### src/features
A series of python files with functions which are called by the `build_datasetV#.py` scripts. This gives the ability 
to capture a variety of different features which are easy to access, the added benefit is that as the competition 
progresses and new features are found it makes adding these new features easily extensible
#### src/model
A series of modelling scripts that call data only to run on, this dir should only include scripts which 
make the final ensemble - investigation should be completed in scratch 


