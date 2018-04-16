# ResourceGather

This project implements an A3C reinforcement learning agent to build bases and gather resources as quickly as possible
using a restricted action space. The project fulfills the requirements of Senior Design (CSC465) at South Dakota School of Mines
and Technology.

## Dependencies

This project requires the following python libraries:

- pysc2 
- numpy
- tensorflow

## Running the Model Enviornment

The test is located in the following folder inside the project directory:

`tests/sc2_ez_env`

The command can be used in this way:

```buildoutcfg
path_to_repo/ResourceGather$ python tests/sc2_ez_env/ez_main.py --help
```
```buildoutcfg
usage: ez_main.py [-h] [--train | --test] [--modelpath dirname] [--gendir]
                  [--hyperparameter-search]
                  [--seed SEED | --seed_range SEED_RANGE SEED_RANGE]

optional arguments:
  -h, --help            show this help message and exit
  --train               If specified, no checkpoint will be loaded, and a new
                        model will be created.
  --test                (Default: False) If specified, a checkpoint will be
                        loaded from modelpath/model if it exists.
  --modelpath dirname   (Default: 'workerData/') If specified, the model and
                        training metrics will be loaded or stored in this
                        location.
  --gendir              If specified, the modelpath will be set to
                        'test_YYYYmmddHHMMSS/'
  --hyperparameter-search
                        If specified, the program will spin up several
                        experiments with different hyperparameters, based on
                        the seed_range.
  --seed SEED           If a hyper parameter search is performed, and a
                        seed_range is not specified, a single run will occur
                        with the hyper-parameters corresponding to that
                        seed. The seed is NOT appended to modelpath
  --seed_range SEED_RANGE SEED_RANGE
                        The range of seeds to use during the hyper-parameter
                        search, the seed is appended to modelpath.
```

## Running the Project

**The main project code will not work in it's current state
To train the Pysc2 Resource Gatherer run


## Running Tests
## Adding a Map
## Documentation
