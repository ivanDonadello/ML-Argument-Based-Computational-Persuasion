"""
Make classification experiments
"""
from DT_simulation import Simulations
import multiprocessing
from sklearn.neural_network import MLPRegressor
from multiprocessing import Process
import os
import numpy as np
import getopt, sys
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from utils import plot_mae
from functools import partial
from custom_regressors import MeanRegressor, RandomRegressor, ClusterRegressor, CRFRegressor
import math
import random
import json
import pdb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, fowlkes_mallows_score
from sklearn.model_selection import KFold
import pandas as pd
import logging
import time
from PlotResults import PlotResults


def multiprocessing_simulations(simulation_setting_dict, tree_population_ids):
    results = {"Tree id": [], "Tree height": [], "Tree leaves": [], "Sample id": [], "Cluster score": [], "Min Lickert value": [], "Max Lickert value": [],
    "Sample clusters": [], "Model": [], "Accuracy": [], "Accuracy std": [], "Mae Prop": [],
    "Mae Prop std": [], "Mae Opp": [], "Mae Opp std": [], "Mean arg distance": [], "Mean arg distance std": []}


    #mae_node_id = {model[0]: {'mae_cv': [], 'std_mae_cv': []} for model in  simulation_setting_dict['models']}
    #mae_Q_proponent = {model[0]: {'mae_cv': [], 'std_mae_cv': []} for model in  simulation_setting_dict['models']}
    #mae_Q_opponent = {model[0]: {'mae_cv': [], 'std_mae_cv': []} for model in  simulation_setting_dict['models']}
    #mean_args_distance = {model[0]: {'mae_cv': [], 'std_mae_cv': []} for model in  simulation_setting_dict['models']}

    tree_id= tree_population_ids[0]
    sample_id = tree_population_ids[1]
    simulation = tree_population_ids[2]
    height = tree_population_ids[3]
    leaves, leaf_names = simulation.get_leaves()


    #for sample_id in range(simulation_setting_dict['number_population_samples']):
    # Random data generation
    num_clusters = random.choice(simulation_setting_dict['clusters'])
    center_box_width = random.choice(simulation_setting_dict['center_box_width'])
    X, y = make_blobs(n_samples=2000, centers=num_clusters, n_features=len(leaves), random_state=0, cluster_std=1, center_box=(-center_box_width,center_box_width))
    X = np.rint(X)
    min_value = np.min(X)
    max_value = np.max(X)
    synth_data_df = pd.DataFrame(X, columns=leaf_names)
    synth_data_df.insert(0, 'id', y, allow_duplicates=True)
    synth_data_df.to_csv(f"data/datasets/tree_{tree_id}_population_{sample_id}.csv", index=False)
    X = synth_data_df.values

    # Check cluster difficulty
    X_train, X_test, y_train, y_test = train_test_split(X[:, 1:], y, test_size=.2, random_state=42)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X_train)
    y_pred = kmeans.predict(X_test)
    fm_score = fowlkes_mallows_score(y_test, y_pred)

    if simulation_setting_dict['debug_mode']:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"acc {accuracy_score(y_test, y_pred)}")

    # Simulation with true utilities (gold standard computation)
    simulation.random_utilities(agent='opp', min_opp=min_value, max_opp=max_value+1)
    output_simulation = simulation.predict(synth_data_df)
    if simulation_setting_dict['debug_mode']:
        cols=['output_node_id', 'Q_proponent_node', 'Q_opponent_node']
        fig, ax = plt.subplots(len(cols), figsize=(15,15))
        output_simulation.astype(int).hist(column=cols, ax=ax)
        fig.savefig('output_histogram.png')
        simulation.to_pdf(f"{tree_id}_{height}")
        output_simulation.to_csv(f"{tree_id}_{height}.csv")
        synth_data_df.to_csv(f"Samples for tree {tree_id}.csv")

    # Iterate over the models
    for name, model, params in simulation_setting_dict['models']:
        logging.info(f"TREE: {tree_id}, SAMPLE: {sample_id}, {name.upper()} model ... ")

        # Model setting
        wrapper_MOR = MultiOutputRegressor(estimator=model, n_jobs=simulation_setting_dict['sklearn_jobs'])
        if name == 'RandomRegressor':
            wrapper_MOR = MultiOutputRegressor(estimator=RandomRegressor(min_value, max_value), n_jobs=simulation_setting_dict['sklearn_jobs'])
        #if name == 'CRFRegressor':
            #model.perform_clustering(X_train, y_train, X_test, y_test)


        # Iterate over the evidence ids
        acc_node_id_mean, mae_Q_proponent_mean, mae_Q_opponent_mean, mean_args_distance_mean = [], [], [], []
        acc_node_id_std, mae_Q_proponent_std, mae_Q_opponent_std, mean_args_distance_std = [], [], [], []
        evidence_list = []
        for evidence_idx, feature_name in enumerate(leaf_names[:-1]):
            print(f"PROCESS: {multiprocessing.current_process().name}, TREE: {tree_id}, SAMPLE: {sample_id},  MODEL {name}, EVIDENCE IDX: {evidence_idx+1}/{len(leaves)}")
            logging.info(f"PROCESS: {multiprocessing.current_process().name}, TREE: {tree_id}, SAMPLE: {sample_id},  MODEL {name}, EVIDENCE IDX: {evidence_idx+1}/{len(leaves)}")
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
                logging.debug(f"Fold {fold_id}, utility prediction with {name}")
                # Selecting samples according to the fold indexes
                train_data = X[train_index, 1:evidence_idx+2]
                train_target = X[train_index, evidence_idx+2:]
                test_data = X[test_index, 1:evidence_idx+2]
                test_target = X[test_index, evidence_idx+2:]


				# Computing the rest of utilities
                if name == 'MLP_keras':
                    wrapper_MOR = KerasRegressor(build_fn=MLP_Model(train_data.shape[1], train_target.shape[1]), epochs=200, batch_size=12, verbose=0)
                    pred_utility = wrapper_MOR.predict(test_data)
                elif name == 'ClusterRegressor':
                    # ClusterRegressor needs the true cluster labels y for computing the best number of clusters
                    # y is unpacked inside the classifier definition
                    #wrapper_MOR.fit(X[train_index, :evidence_idx+2], train_target)
                    #pdb.set_trace()
                    model.fit(X[train_index, :evidence_idx+2], train_target)
                    #pdb.set_trace()
                    pred_utility = model.predict(test_data)
                elif name == 'CRFRegressor':
                    cluster_labels = model.perform_clustering(X[train_index])
                    #cluster_labels = model.perform_clustering(X[train_index, 1:], y[train_index], X[test_index, 1:], y[test_index])
                    model.fit((X[train_index, 1:], evidence_idx+2), cluster_labels)
                    #model.fit((X[train_index, 1:], evidence_idx+1), y[train_index])
                    _, pred_utility, _ = model.predict(X[test_index, 1:])
                else:
                    wrapper_MOR.fit(train_data, train_target)
                    pred_utility = wrapper_MOR.predict(test_data)

				# Selecting true data from simulations with true utility
                output_simulation_test = output_simulation.iloc[test_index]

				# Combining the predicted utility with the rest of the user's evidence/data
                selected_samples = synth_data_df.iloc[test_index].copy()
                for idx, col_name in enumerate(leaf_names[evidence_idx+1:]):
                    if name == 'MLP_keras':
                        selected_samples[col_name] = np.rint(pred_utility)
                        #selected_samples[col_name] = pred_utility
                    elif name == 'CRFRegressor':
                        pass
                    else:
                        selected_samples[col_name] = np.rint(pred_utility[:, idx])
                        #selected_samples[col_name] = pred_utility[:, idx]
                if name == 'CRFRegressor':
                    #selected_samples_tmp = selected_samples.copy()
                    selected_samples[leaf_names] = np.rint(pred_utility)
                    #selected_samples[leaf_names] = pred_utility

				# Bimaximax simulation with predicted utility
                output_simulation_pred_utility = simulation.predict(selected_samples)

                # Store the results of each fold
                acc_node_id_fold.append(accuracy_score(output_simulation_test['output_node_id'].to_numpy(), output_simulation_pred_utility['output_node_id'].to_numpy()))
                mae_Q_proponent_fold.append(mean_absolute_error(output_simulation_test['Q_proponent_node'].to_numpy(), output_simulation_pred_utility['Q_proponent_node'].to_numpy()))
                mae_Q_opponent_fold.append(mean_absolute_error(output_simulation_test['Q_opponent_node'].to_numpy(), output_simulation_pred_utility['Q_opponent_node'].to_numpy()))
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
            logging.info(f"\tAccuracy node {acc_node_id_mean[-1]:.2f} +- {acc_node_id_std[-1]:.2f}")
            logging.info(f"\tMae Q prop {mae_Q_proponent_mean[-1]:.2f} +- {mae_Q_proponent_std[-1]:.2f}")
            logging.info(f"\tMae Q opp {mae_Q_opponent_mean[-1]:.2f} +- {mae_Q_opponent_std[-1]:.2f}")
            logging.info(f"\tMean argument distance {mean_args_distance_mean[-1]:.2f} +- {mean_args_distance_mean[-1]:.2f}")
            print(f"\tAccuracy node {acc_node_id_mean[-1]:.2f} +- {acc_node_id_std[-1]:.2f}")
            print(f"\tMae Q prop {mae_Q_proponent_mean[-1]:.2f} +- {mae_Q_proponent_std[-1]:.2f}")
            print(f"\tMae Q opp {mae_Q_opponent_mean[-1]:.2f} +- {mae_Q_opponent_std[-1]:.2f}")
            print(f"\tMean argument distance {mean_args_distance_mean[-1]:.2f} +- {mean_args_distance_std[-1]:.2f}")


        # Store results after the computation over all evidence
        results["Tree id"].append(tree_id)
        results["Tree height"].append(height)
        results["Tree leaves"].append(len(leaves))
        results["Sample id"].append(sample_id)
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
    return results


if __name__ == '__main__':
    # Keep all but the first
    argument_list = sys.argv[1:]

    try:
        arguments, values = getopt.getopt(argument_list, "p:", ["parallel"])
    except getopt.error as err:
        # Output error, and return with an error code
        print(str(err))
        sys.exit(2)

    for opt, arg in arguments:
        if opt == '-p':
            use_pool_multiproc = True if arg == 'True' else False
            sklearn_jobs = 1 if arg == 'True' else -1

    # ML models for utility learning
    models = [('SuppVecMachTuned', SVR(C=0.5, gamma='auto'), {'estimator__gamma': ['scale', 'auto'], 'estimator__C': [0.1, 1, 10],'estimator__kernel': ['rbf']}),
    ('ClusterRegressor', ClusterRegressor(num_clusters=[4, 6, 8, 10]), {}),('MeanRegressor', MeanRegressor(), {}),
	            ('RandomRegressor', RandomRegressor(), {}), ('SuppVecMach', SVR(), {'estimator__gamma': ['scale', 'auto'], 'estimator__C': [0.1, 1, 10],'estimator__kernel': ['rbf']})]

    models = [('ClusterRegressor', ClusterRegressor(num_clusters=[4, 6, 8, 10]), {}),
    ('CRFRegressor', CRFRegressor(num_clusters_list=[4, 6, 8, 10]), {}),
    ('MeanRegressor', MeanRegressor(), {}),
    ('RandomRegressor', RandomRegressor(), {}),
    ('SuppVecMach', SVR(), {'estimator__gamma': ['scale', 'auto'], 'estimator__C': [0.1, 1, 10],'estimator__kernel': ['rbf']})]
    #models = [('CRFRegressor', CRFRegressor(num_clusters_list=[4, 6, 8, 10]), {}),
    #('MeanRegressor', MeanRegressor(), {}), ('RandomRegressor', RandomRegressor(), {})]
    # simulation settings
    JOBS = multiprocessing.cpu_count()
    debug_mode = False
    RESULTS_PATH = 'results'
    simulation_setting_dict = {'height_simulated_trees': [4, 5, 6], 'sklearn_jobs': sklearn_jobs,
    'branching_factors': [2, 3, 4], 'models': models, 'debug_mode': debug_mode,
    'clusters': [4, 6, 8, 10], 'center_box_width': [0.5, 2.5, 1, 3]}
    tree_number = 10
    number_population_samples = 10
    info_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(filename='results/simulation_experiments.log', filemode='w', format=f'%(asctime)s-%(levelname)s-%(message)s', datefmt='%d-%b-%y %H:%M:%S', level=info_level)


    # Trees and population ids generation
    logging.info("Trees and population ids generation")
    tree_population_ids_list = []
    for tree_id in range(tree_number):
        simulation_ = Simulations()
        height_ = random.choice(simulation_setting_dict['height_simulated_trees'])
        simulation_.generate_random_tree(height_, simulation_setting_dict['branching_factors'])
        simulation_.random_utilities(agent='prop')
        simulation_.root.compute_chance_decision(True)
        simulation_.to_csv(f"data/DT/tree_{tree_id}.csv")
        for sample_id in range(number_population_samples):
            tree_population_ids_list.append((tree_id, sample_id, simulation_, height_))

    final_results = []
    start_time = time.time()
    if use_pool_multiproc:
        pool = multiprocessing.Pool(processes=JOBS)
        func = partial(multiprocessing_simulations, simulation_setting_dict)
        final_results = pool.map(func, tree_population_ids_list)
        pool.close()
    else:
        for el in tree_population_ids_list:
            final_results.append(multiprocessing_simulations(simulation_setting_dict, el))
    print(f"Simulations took {(time.time() - start_time)/3600.} hours")

    results = final_results[0].copy()
    for id in range(1, len(final_results)):
        for k in results.keys():
            results[k] += final_results[id][k]

    results_df = pd.DataFrame.from_dict(results)
    plot_res = PlotResults(results_df, RESULTS_PATH)
    plot_res.plot_single_trees()
    plot_res.plot_aggregate_results('Tree leaves')
    plot_res.plot_aggregate_results('Sample clusters')
    plot_res.plot_aggregate_results('Cluster score')
    plot_res.plot_aggregate_results_evidence_perc()
    results_df.to_csv(os.path.join(RESULTS_PATH, 'results.csv'), index=False)
