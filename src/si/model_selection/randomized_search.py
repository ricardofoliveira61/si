import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
import itertools
from si.model_selection.cross_validate import k_fold_cross_validation


def randomized_search_cv(model:Model,dataset:Dataset,hyperparameter_grid:dict,
                        cv:int,n_iter:int,scoring:callable = None)->dict:
    
    """
    Performs randomized grid search
    Randomized Grid Search is a hyperparameter tuning technique that explores a specified
    parameter space by randomly sampling combinations of hyperparameter values.
    This approach is often more efficient than a traditional grid search, especially
    when dealing with a large number of hyperparameters.
    The main advantage of this approach are:
    - Efficiency: Can explore a larger hyperparameter space more efficiently than grid search.
    - Flexibility: Allows for continuous or discrete hyperparameter spaces.
    - Handles High-Dimensional Spaces: Can be effective for models with many hyperparameters.
    

    Parameters :
    ----------
    model : Model
        - The model to perform the hyperparameter tunning
    dataset : Dataset
        - The validation dataset to perform the hyperparameter tunning
    hyperparameter_grid : dict
        - Dictionary with the hyperparameter name and search values
    scoring : callable
        - The scoring function to evaluate the model's performance during the hyperparameter tunning
    cv : int
        - Number of folds
    n_iter : int
        - Number of hyperparameter random combinations to search

    Returns :
    ----------
    results : dict
        - Dictionary with the results of the grid search cross validation. Includes the
        scores, hyperparameters, best hyperparameters and best score
    """
    
    # check if the hyperparameter are present in the model
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")
    
    # select n_iter random combinations from all combinations possible for the hyperparameters
    combinations = random_combinations(hyperparameter_grid = hyperparameter_grid, n_iter=n_iter)

    # initializing the results dictionary
    results = {'scores': [], 'hyperparameters': []}

    for combination in combinations:

        # parameter configuration
        parameters = {}

        # set the parameters
        for parameter, value in zip(hyperparameter_grid.keys(), combination):
            setattr(model, parameter, value)
            parameters[parameter] = value

        # cross validate the model
        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # add the score
        results['scores'].append(np.mean(score))

        # add the hyperparameters
        results['hyperparameters'].append(parameters)

    results['best_hyperparameters'] = results['hyperparameters'][np.argmax(results['scores'])]
    results['best_score'] = np.max(results['scores'])
    return results


def random_combinations(hyperparameter_grid:dict,n_iter:int)->list:
    """
    From all the possible hyperparameters combinatios selects randomly n_iter combinations of hyperparameters

    Parameters :
    -----------
    hyperparameter_grid: dict
        - Dictionary with the hyperparameter name and search values
    n_iter: int
        - Number of combinations to randomly select
    
    Returns :
    -----------
    random_combinations: list
        - List of the random combinations of hyperparameters
    """

    # computing all combinations of hyperparameters possible
    all_combinations = list(itertools.product(*hyperparameter_grid.values()))
    # select random indices form all combinations
    random_indices = np.random.choice(len(all_combinations),n_iter, replace=False)
    # select the random combinations from all combinations
    random_combinations = [all_combinations[idx] for idx in random_indices]

    return random_combinations
