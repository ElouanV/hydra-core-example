# hydra-core-example
It does not go deep into the details and so this example does not exploit all the potential of hydra, only the most accessible part. `hydra-core` can be plugged to a lot of other ML package such as Optuna, to define hyperparameter sweeps and perform hyperparameters optimization without changing the code of the project... and a lot more :)
## Installation
`pip install hydra-core --upgrade`

For this example:

`pip install -r requirements.txt`
## Run examples
```sh
python main.py models=logistic_regression datasets=wine
```

will run the script with overriding the parameter `models` and the parameter `datasets`




```sh
python main.py --multirun models=logistic_regression datasets=wine 'models.param.wine.C=range(0.1,1.0,0.1)'
```

This will run an hyperparameter optimization on C parameter of the logistic regression for dataset wine.
It will create a folder `multirun/` in which you will find logs of all runs and a `optimization_results.yaml` that summarize the result of the Optuna hyperparameters research.

Python `logging` library is used and configured automatically by Hydra using `hydra/job_logging/custom.yaml`



## Hydra core
### Documentation
[Docs](https://hydra.cc/docs/intro/)
### Repository
[GitHub](https://github.com/facebookresearch/hydra)

### Description
Hydra is an open-source Python framework that simplifies the development of research and other complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line. The name Hydra comes from its ability to run multiple similar jobs - much like a Hydra with multiple heads.
### Key features
- Hierarchical configuration composable from multiple sources
- Configuration can be specified or overridden from the command line
- Dynamic command line tab completion
- Run your application locally or launch it to run remotely
- Run multiple jobs with different arguments with a single command

## Some repositories that use hydra-core
* [Nvidia](https://github.com/NVIDIA/DeepLearningExamples/tree/master)
* [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template/tree/main)
* [GStarX](https://github.com/ShichangZh/GStarX)
* [Tune(HuggingFace)](https://github.com/huggingface/tune/tree/main)

## Other
[Meta blog post](https://ai.meta.com/blog/reengineering-facebook-ais-deep-learning-platforms-for-interoperability/)