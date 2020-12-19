from sklearn.model_selection import KFold


def cross_validation(args, train_data, test_data=None):
    """
    Function to run cross validation to train a model.
    :param args: TrainArgs object containing the parameters for the cross-validation run.
    :param train_data: The training to be used in cross-validation
    :param test_data: The testing data if loaded to be used to evaluate model performance.
    :return:
    """
    folds = args.cv_folds
    base_dir = args.save_dir
    seed = args.seed

    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    train_examples = list(train_data.x.size())[1]
    print(train_examples)

    #for train_index, valid_index in kfold.split()