import pandas as pd
import pdb
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import math

plt.rcParams.update({'lines.linewidth': 2.0})


def str2list(input_list):
	output_list = input_list
	if type(input_list) is str:
		splitted_string = re.split('\[|\]|,\s', input_list)
		return [float(el) for el in splitted_string if el != '']
	else:
		return output_list


def set_axis_plot(title_, prefix="", x_label=""):
	fig = plt.figure(figsize=(18, 5)) #(24, 5) if accuracy
	#ax1 = fig.add_subplot(141)
	ax2 = fig.add_subplot(131)
	ax3 = fig.add_subplot(132)
	ax4 = fig.add_subplot(133)
	fig.suptitle(f'{title_}', fontsize=16)
	#ax1.set_title(f"{prefix}accuracy", fontsize=18)
	ax2.set_title(f"{prefix}mean argument distance", fontsize=18)
	ax3.set_title(f"{prefix}mean abs. error proponent utility", fontsize=18)
	ax4.set_title(f"{prefix}mean abs. error opponent utility", fontsize=18)
	#ax1.set_xlabel(x_label, fontsize=18)
	ax2.set_xlabel(x_label, fontsize=18)
	ax3.set_xlabel(x_label, fontsize=18)
	ax4.set_xlabel(x_label, fontsize=18)
	return fig, ax2, ax3, ax4


class PlotResults():
	def __init__(self, data, folder):
		self.folder = folder
		self.data = data
		self.metrics = ['Mean arg distance', 'Mae Prop', 'Mae Opp', 'Mean arg distance std', 'Mae Prop std', 'Mae Opp std']
		#self.metrics = ['Accuracy', 'Mean arg distance', 'Mae Prop', 'Mae Opp', 'Accuracy std', 'Mean arg distance std', 'Mae Prop std', 'Mae Opp std']
		self.num_metrics = int(len(self.metrics)/2)
		self.method_label = {'ClusterRegressor': r'$\mathrm{CLUMER}$', 'CRFRegressor':r'$\mathrm{CRAMER}$', 'MeanRegressor':r'$\mathrm{MeanR}$', 'RandomRegressor': r'$\mathrm{RandR}$', 'SuppVecMach':r'$\mathrm{SVR}$', 'GradientBoost': r'$\mathrm{GBOOST}$'}
		self.method_marker = {'ClusterRegressor': '<', 'CRFRegressor':'x', 'MeanRegressor':'s', 'RandomRegressor': '>', 'SuppVecMach':'D', 'GradientBoost': '^'}
		self.method_color = {'ClusterRegressor': 'mediumpurple', 'CRFRegressor':'deepskyblue', 'MeanRegressor':'orange', 'RandomRegressor': 'crimson', 'SuppVecMach':'forestgreen', 'GradientBoost': 'blue'}
		self.method_line = {'ClusterRegressor': (0, (3, 1, 1, 1)), 'CRFRegressor':(0, (5, 1)), 'MeanRegressor':(0, (3, 5, 1, 5, 1, 5)), 'RandomRegressor': 'dotted', 'SuppVecMach':'-', 'GradientBoost': '--'}



	def plot_single_trees(self):
		for tree_id in pd.unique(self.data['Tree id']):
			for sample_id in pd.unique(self.data['Sample id']):
				title = f"Tree id {tree_id}, Sample id {sample_id}"
				title=''
				plt.clf()
				fig, ax2, ax3, ax4 = set_axis_plot(title, x_label='Number questions asked') # TODO ax1 if accuracy
				query_res = self.data.loc[(self.data['Tree id'] == tree_id) & (self.data['Sample id'] == sample_id)]
				for idx, single_run in query_res.iterrows():
					evidence = [el+1 for el in range(single_run['Tree leaves']-1)]
					#ax1.errorbar(evidence, str2list(single_run['Accuracy']), yerr=str2list(single_run['Accuracy std']), linestyle=self.method_line[single_run['Model']], color=self.method_color[single_run['Model']], marker=self.method_marker[single_run['Model']], label=self.method_label[single_run['Model']])
					ax2.errorbar(evidence, str2list(single_run['Mean arg distance']), yerr=str2list(single_run['Mean arg distance std']), linestyle=self.method_line[single_run['Model']], color=self.method_color[single_run['Model']], marker=self.method_marker[single_run['Model']], label=self.method_label[single_run['Model']])
					ax3.errorbar(evidence, str2list(single_run['Mae Prop']), yerr=str2list(single_run['Mae Prop std']), linestyle=self.method_line[single_run['Model']], color=self.method_color[single_run['Model']], marker=self.method_marker[single_run['Model']], label=self.method_label[single_run['Model']])
					ax4.errorbar(evidence, str2list(single_run['Mae Opp']), yerr=str2list(single_run['Mae Opp std']), linestyle=self.method_line[single_run['Model']], color=self.method_color[single_run['Model']], marker=self.method_marker[single_run['Model']], label=self.method_label[single_run['Model']])
				ax4.legend(fontsize=14)
				if not os.path.exists(os.path.join(self.folder, 'tree_samples')):
					os.makedirs(os.path.join(self.folder, 'tree_samples'))
				plt.tight_layout()
				title='geppino'
				plt.savefig(os.path.join(self.folder, 'tree_samples', f'{title}.pdf'))


	def plot_aggregate_results(self, aggregation_param):
		title = f"Results on {aggregation_param.lower()} aggregation"
		fig, ax2, ax3, ax4 = set_axis_plot(title, prefix="Average ", x_label=f'Number of {aggregation_param.lower()}') # TODO ax1 if accuracy
		axis_list = [ax2, ax3, ax4] # TODO ax1 if accuracy
		aggr_dict = {model_name: {el: [] for el in self.metrics} for model_name in self.data["Model"].unique()}
		x_axis_values = []
		unique_value = self.data[aggregation_param].unique()
		for aggreg_value in sorted(unique_value):
			x_axis_values.append(aggreg_value)
			query_res = self.data.loc[(self.data[aggregation_param] == aggreg_value)]
			aggr_dict_value = {model_name: {el: [] for el in self.metrics} for model_name in self.data["Model"].unique()}

			for idx, single_run in query_res.iterrows():
				for metric in self.metrics:
					aggr_dict_value[single_run['Model']][metric] += str2list(single_run[metric])

			for model_name in aggr_dict.keys():
				for metric in self.metrics:
					if metric.split(" ")[-1] == 'std':
						# sampled variance
						aggr_dict[model_name][metric].append(np.sqrt(np.mean(np.square(aggr_dict_value[model_name][metric]))))
					else:
						aggr_dict[model_name][metric].append(np.mean(aggr_dict_value[model_name][metric]))

		for model_name in aggr_dict.keys():
			for idx, metric in enumerate(self.metrics[:self.num_metrics]):
				tmp_ax = axis_list[idx]
				tmp_ax.errorbar(x_axis_values, aggr_dict[model_name][metric], yerr=aggr_dict[model_name][f"{metric} std" ], color=self.method_color[model_name], marker=self.method_marker[model_name], label=self.method_label[model_name])
		ax4.legend(fontsize=14)
		plt.tight_layout()
		plt.savefig(os.path.join(self.folder, f'{title}.pdf'))


	def plot_aggregate_results_evidence_perc(self):
		title = ""
		fig, ax2, ax3, ax4 = set_axis_plot(title, prefix="Average ", x_label='Percentage of evidence') # TODO ax1 if accuracy
		axis_list = [ax2, ax3, ax4] # TODO ax1 if accuracy
		x_axis_values = [(el+1)*10 for el in range(10)]
		aggr_dict = {model_name: {el: [] for el in self.metrics} for model_name in self.data["Model"].unique()}
		for evidence_perc in x_axis_values:
			aggr_dict_value = {model_name: {el: [] for el in self.metrics} for model_name in self.data["Model"].unique()}
			for idx, single_run in self.data.iterrows():
				for metric in self.metrics:
					results_list = str2list(single_run[metric])
					selected_values = math.ceil(len(results_list)*evidence_perc/100)
					aggr_dict_value[single_run["Model"]][metric] += str2list(single_run[metric])[:selected_values]

			for model_name in aggr_dict.keys():
				for metric in self.metrics:
					if metric.split(" ")[-1] == 'std':
						# sampled variance
						aggr_dict[model_name][metric].append(np.sqrt(np.mean(np.square(aggr_dict_value[model_name][metric]))))
					else:
						aggr_dict[model_name][metric].append(np.mean(aggr_dict_value[model_name][metric]))
		for model_name in aggr_dict.keys():
			for idx, metric in enumerate(self.metrics[:self.num_metrics]):
				tmp_ax = axis_list[idx]
				tmp_ax.errorbar(x_axis_values, aggr_dict[model_name][metric], yerr=aggr_dict[model_name][f"{metric} std" ], linestyle=self.method_line[model_name], color=self.method_color[model_name], marker=self.method_marker[model_name], label=self.method_label[model_name])
		ax4.legend(fontsize=14)
		plt.tight_layout()
		title = f"Results on percentage of evidence aggregation"
		plt.savefig(os.path.join(self.folder, f'{title}.pdf'))


if __name__ == '__main__':
	results_df = pd.read_csv('meat_results/results.csv')
	plot_res = PlotResults(results_df, "meat_results")
	#results_df = pd.read_csv('results/results.csv')
	#plot_res = PlotResults(results_df, "results")
	plot_res.plot_single_trees()
	#plot_res.plot_aggregate_results('Tree leaves')
	#plot_res.plot_aggregate_results('Sample clusters')
	#plot_res.plot_aggregate_results_evidence_perc()
