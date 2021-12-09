import utils
import os
from sklearn.metrics import mean_absolute_error, accuracy_score, fowlkes_mallows_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pdb
import pandas as pd
from PlotResults import PlotResults
from DT_simulation import Simulations
import csv
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from custom_regressors import MeanRegressor, RandomRegressor, ClusterRegressor, CRFRegressor
import matplotlib.pyplot as plt


if __name__ == '__main__':
    SEED = 0
    DATA_PATH = "meat_data"
    RESULTS_PATH = 'meat_results'
    id_demographic_attributes = 5
    data_variance = 1
    frontier_argument_ids = np.genfromtxt(os.path.join(DATA_PATH, 'frontier_arguments.csv'), delimiter=',', dtype=np.int32, usecols=[0])
    profiles = np.genfromtxt(os.path.join(DATA_PATH, 'profiles.csv'), delimiter=',', names=True, dtype=None)
    # Load/create dataset
    #X = np.genfromtxt(os.path,join(DATA_PATH, 'meat_args_dataset_var_1.csv'))
    X, whole_samples_df, evidence_names = utils.build_synt_data_from_profiles(frontier_argument_ids, profiles, id_demographic_attributes+1)
    #pdb.set_trace()
    whole_samples_df.to_csv(os.path.join(DATA_PATH, f"meat_args_dataset_var_{data_variance}.csv"), index=False)
    with open(os.path.join(DATA_PATH, 'evidence_names.csv'), 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(evidence_names)
    evidence_idx_list = [*range(id_demographic_attributes, X[:, 1:].shape[1])]
    min_value = np.min(X[:, 1:])
    max_value = np.max(X[:, 1:])
    num_clusters = len(np.unique(X[:, 0]))

    # Check cluster difficulty
    X_train, X_test, y_train, y_test = train_test_split(X[:, 1:], X[:, 0], test_size=.2, random_state=SEED)
    kmeans = KMeans(n_clusters=num_clusters, random_state=SEED).fit(X_train)
    y_pred = kmeans.predict(X_test)
    fm_score = fowlkes_mallows_score(y_test, y_pred)

    # Results dict
    results = {"Tree id": [], "Tree height": [], "Tree leaves": [], "Sample id": [], "Cluster score": [], "Min Lickert value": [], "Max Lickert value": [],
    "Sample clusters": [], "Model": [], "Accuracy": [], "Accuracy std": [], "Mae Prop": [],
    "Mae Prop std": [], "Mae Opp": [], "Mae Opp std": [], "Mean arg distance": [], "Mean arg distance std": []}

    # Initialization of bimaximax simulations
    simulation = Simulations()
    simulation.from_csv(os.path.join(DATA_PATH, "meat_DT.csv"))
    simulation.prop_utilities_from_json(os.path.join(DATA_PATH, "meat_DT_extra.json"))
    simulation.root.compute_chance_decision(True)
    height = 4
    leaves = simulation.get_leaves()[1]

    # Offline computing of the simulations with the true utility
    output_simulation = simulation.predict(whole_samples_df, use_user_model=False)

    # Simulation statistics
    cols=['Node id', 'Proponent utility', 'Opponent utility']
    fig, ax = plt.subplots(len(cols), figsize=(15,15))
    #fig, ax = plt.subplots(1, figsize=(15,15))
    #ax.set_xticks(np.arange(10, 160, 1))
    output_simulation.astype(int).hist(column=cols, bins=30, ax=ax)
    #output_simulation.astype(int).hist(column=['Node id'], bins=30, ax=ax, xrot=90)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_PATH, 'meat_simualtions_hist.png'))

    models = [('CRFRegressor', CRFRegressor(num_clusters_list=[5, 6, 8, 10]), {}), ('RandomRegressor', RandomRegressor(min_value, max_value), {}),
    ('SuppVecMach', SVR(), {'estimator__gamma': ['scale', 'auto'], 'estimator__C': [0.1, 1, 10],'estimator__kernel': ['rbf']}),('MeanRegressor', MeanRegressor(), {}),
    ('ClusterRegressor', ClusterRegressor(num_clusters=[5, 6, 8, 10]), {'estimator__gamma': ['scale', 'auto'], 'estimator__C': [0.1, 1, 10],'estimator__kernel': ['rbf']})]


    # Iterate over the models
    for name, model, params in models:
        # Model setting
        wrapper_MOR = MultiOutputRegressor(estimator=model, n_jobs=-1)

        # Iterate over the evidence ids
        acc_node_id_mean, mae_Q_proponent_mean, mae_Q_opponent_mean, mean_args_distance_mean = [], [], [], []
        acc_node_id_std, mae_Q_proponent_std, mae_Q_opponent_std, mean_args_distance_std = [], [], [], []
        evidence_list = []
        for evidence_idx in evidence_idx_list:
            print(f"MODEL: {name}, EVIDENCE IDX: {evidence_idx}")
            evidence_list.append(evidence_idx)

            # Setting K-fold variables
            kf = KFold()
            kf.get_n_splits(X)
            acc_node_id_fold = []
            mae_Q_proponent_fold = []
            mae_Q_opponent_fold = []
            mean_args_distance_fold = []
            fold_id = 0

            for train_index, test_index in kf.split(X):
                # Selecting samples according to the fold indexes
                
                train_data = X[train_index, 1:evidence_idx+1]
                train_target = X[train_index, evidence_idx+1:]
                test_data = X[test_index, 1:evidence_idx+1]
                test_target = X[test_index, evidence_idx+1:]

                # Computing the rest of utilities
                if name == 'MLP_keras':
                    wrapper_MOR = KerasRegressor(build_fn=MLP_Model(train_data.shape[1], train_target.shape[1]), epochs=200, batch_size=12, verbose=0)
                    pred_utility = wrapper_MOR.predict(test_data)
                elif name == 'ClusterRegressor':
                    # ClusterRegressor needs the true cluster labels y for computing the best number of clusters
                    # y is unpacked inside the classifier definition
                    model.fit(X[train_index, :evidence_idx+1], train_target)
                    pred_utility = model.predict(test_data)
                elif name == 'CRFRegressor':
                    cluster_labels = model.perform_clustering(X[train_index])
                    model.fit((X[train_index, 1:], evidence_idx), cluster_labels)
                    #model.fit((X[train_index, 1:], evidence_idx-id_demographic_attributes+2), cluster_labels)
                    _, pred_utility, _ = model.predict(X[test_index, 1:])
                else:
                    wrapper_MOR.fit(train_data, train_target)
                    pred_utility = wrapper_MOR.predict(test_data)

                # Selecting true data from simulations with true utility
                output_simulation_test = output_simulation.iloc[test_index]

				# Combining the predicted utility with the rest of the user's evidence/data
                selected_samples = whole_samples_df.iloc[test_index].copy()
                for idx, col_name in enumerate(evidence_names[evidence_idx+1:]):
                    if name == 'MLP_keras':
                        selected_samples[col_name] = np.rint(pred_utility)
                        #selected_samples[col_name] = pred_utility
                    elif name == 'CRFRegressor':
                    	pass
                    else:
                        selected_samples[col_name] = np.rint(pred_utility[:, idx])
                        #selected_samples[col_name] = pred_utility[]
                if name == 'CRFRegressor':
                	selected_samples[evidence_names[id_demographic_attributes+1:]] = np.rint(pred_utility[:, id_demographic_attributes:])
                    #selected_samples[leaf_names] = np.rint(pred_utility)

				# Bimaximax simulation with predicted utility
                #pdb.set_trace()
                output_simulation_pred_utility = simulation.predict(selected_samples, use_user_model=False)

                # Store the results of each fold
                acc_node_id_fold.append(accuracy_score(output_simulation_test['Node id'].to_numpy(), output_simulation_pred_utility['Node id'].to_numpy()))
                mae_Q_proponent_fold.append(mean_absolute_error(output_simulation_test['Proponent utility'].to_numpy(), output_simulation_pred_utility['Proponent utility'].to_numpy()))
                mae_Q_opponent_fold.append(mean_absolute_error(output_simulation_test['Opponent utility'].to_numpy(), output_simulation_pred_utility['Opponent utility'].to_numpy()))
                mean_args_distance_fold.append(mae_Q_proponent_fold[-1] + mae_Q_opponent_fold[-1]) # taxicab distance between arguements
                fold_id += 1

            # Aggregate fold results after the KFold computation
            acc_node_id_mean.append(np.mean(acc_node_id_fold))
            acc_node_id_std.append(np.std(acc_node_id_fold))
            mae_Q_proponent_mean.append(np.mean(mae_Q_proponent_fold))
            mae_Q_proponent_std.append(np.std(mae_Q_proponent_fold))
            mae_Q_opponent_mean.append(np.mean(mae_Q_opponent_fold))
            mae_Q_opponent_std.append(np.std(mae_Q_opponent_fold))
            mean_args_distance_mean.append(np.mean(mean_args_distance_fold))
            mean_args_distance_std.append(np.std(mean_args_distance_fold))
            print(f"\tAccuracy node {acc_node_id_mean[-1]:.2f} +- {acc_node_id_std[-1]:.2f}")
            print(f"\tMae Q prop {mae_Q_proponent_mean[-1]:.2f} +- {mae_Q_proponent_std[-1]:.2f}")
            print(f"\tMae Q opp {mae_Q_opponent_mean[-1]:.2f} +- {mae_Q_opponent_std[-1]:.2f}")
            print(f"\tMean argument distance {mean_args_distance_mean[-1]:.2f} +- {mean_args_distance_std[-1]:.2f}")

        # Store results after the computation over all evidence
        results["Tree id"].append(0)
        results["Tree height"].append(height)
        results["Tree leaves"].append(len(evidence_idx_list)+1)
        results["Sample id"].append(0)
        results["Cluster score"].append(round(fm_score, 2))
        results["Min Lickert value"].append(min_value)
        results["Max Lickert value"].append(max_value)
        results["Sample clusters"].append(num_clusters)
        results["Model"].append(name)
        results["Accuracy"].append(acc_node_id_mean)
        results["Accuracy std"].append(acc_node_id_std)
        results["Mae Prop"].append(mae_Q_proponent_mean)
        results["Mae Prop std"].append(mae_Q_proponent_std)
        results["Mae Opp"].append(mae_Q_opponent_mean)
        results["Mae Opp std"].append(mae_Q_opponent_std)
        results["Mean arg distance"].append(mean_args_distance_mean)
        results["Mean arg distance std"].append(mean_args_distance_std)

    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(os.path.join(RESULTS_PATH, 'results.csv'), index=False)
    plot_res = PlotResults(results_df, RESULTS_PATH)
    plot_res.plot_single_trees()
