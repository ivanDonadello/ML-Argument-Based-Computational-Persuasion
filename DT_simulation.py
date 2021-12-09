from operator import itemgetter
from tree_node_DT import TreeNode
import random
import numpy as np
import pandas as pd
import csv
import json


def render_text(input_string):
    space_idxs = [m.start() for m in re.finditer(' ', input_string)]
    for i in range(len(space_idxs)):
        if (i + 1)%6 ==0:
            input_string = input_string[: space_idxs[i]] + '\n' + input_string[space_idxs[i] + 1:]
    return input_string


class Simulations:
    """
    Class for simulating a dialogue
    """
    def __init__(self):
        self.dict_tree = None
        self.root = None
        self.node_results = []
        self.user_model = {}
        self.extra_data = {}


    def reset_opponent_utilities(self):
        for _, node in self.dict_tree.items():
            node.Q_opponent = -1
            node.Q_proponent = -1
            node.utility_opponent = -1


    def reset_results(self):
        self.node_results = []

    def reset_user_model(self):
        self.user_model = {}

    def set_user_model(self, sample):
        #Qui ci vuole il filename sugli argomenti interni di ongi nodo :/ va preso dai sample....
        # dato un nodo id torna quello del figlio in base al profilo. Serve anche il same as e il nome delle colonne.
        # se N figli hanno stessa utilità prendi il minore, evita effetti random.
        for arg_id in self.extra_data["inner"]:
            self.user_model[arg_id] = int(sample[arg_id])
            if arg_id in self.extra_data["same_as"]:
                for same_arg in self.extra_data["same_as"][arg_id]:
                    self.user_model[same_arg] = int(sample[arg_id])


    def to_pdf(self, name):
        g = Digraph('G', filename=f'profile_{name}.gv')
        g.attr(rankdir='RL', size='8,5')
        for _, tmp_node in self.dict_tree.items():
            g.attr('node', shape='box', style='filled')
            g.node(tmp_node.id, render_text(str(tmp_node)))
        for child in tmp_node.children:
            g.node(child.id, str(child))
            g.edge(child.id, tmp_node.id)
        g.view()
        
        
    def to_csv(self, filename):
        with open(filename, mode='w') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(['Node_id', 'Type', 'Children_ids', 'Utility_proponent', 'Utility_opponent'])
            for node_id, node in self.dict_tree.items():
                writer.writerow([node_id, node.text, [child.id for child in node.children], node.utility_proponent, node.utility_opponent])


    def get_leaves(self):
        leaf_list = []
        leaf_names_list = []
        for _, node in self.dict_tree.items():
            if node.isLeaf():
                leaf_list.append(node)
                leaf_names_list.append(node.id)
        return leaf_list, leaf_names_list


    def generate_random_tree(self, max_height, branching_factors):
        self.dict_tree = {}
        #self.root = None
        last_node_id = 0
        self.root = TreeNode('0', 'root')
        self.root.height = 0
        self.dict_tree['0'] = self.root

        persuasion_goal_node = TreeNode('1', 'persuasion goal')
        persuasion_goal_node.height = 0
        last_node_id = 1
        self.dict_tree['1'] = persuasion_goal_node
        self.root.add_child(persuasion_goal_node)

        stack = []
        discovered = []
        stack.append(persuasion_goal_node)
        while len(stack) > 0:
            tmp_node = stack.pop()
            if tmp_node.id not in discovered:
                discovered.append(tmp_node)
                # low weight to nodes with 4 branches to avoid trees with too many leaves
                branch_factor = np.random.choice(branching_factors, p=[0.45, 0.45, 0.1])
                for id_offset in range(branch_factor):
                    last_node_id += 1
                    id_new_child = str(last_node_id)
                    new_child = TreeNode(id_new_child, "")
                    new_child.height = tmp_node.height + 1
                    tmp_node.add_child(new_child)
                    self.dict_tree[id_new_child] = new_child
                    if new_child.height < max_height-1:
                        stack.append(new_child)
        _, leaf_ids = self.get_leaves()
        self.extra_data["frontier"] = leaf_ids
        self.extra_data["same_as"] = []


    def from_csv(self, filename):
        self.dict_tree = {}
        #self.root = None
        self.root = TreeNode('0', 'root')
        self.dict_tree['0'] = self.root
        with open(filename) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                tmp_node = TreeNode(row['id'], row['text'])
                self.dict_tree[row['id']] = tmp_node
                if row['support'] == '' and row['attack'] == '': # first node after root
                    self.root.add_child(tmp_node)
                elif row['support'] != '':
                    self.dict_tree[row['support']].add_child(tmp_node)
                else:
                    self.dict_tree[row['attack']].add_child(tmp_node)


    def random_utilities(self, agent='both', min_opp=1, max_opp=11, min_prop=1, max_prop=11):
        for id, node in self.dict_tree.items():
            if node.isLeaf():
                if agent == 'both':
                    node.utility_opponent = random.randint(min_opp, max_opp)
                    node.utility_proponent = random.randint(min_prop, max_prop)
                elif agent == 'opp':
                    node.utility_opponent = random.randint(min_opp, max_opp)
                else:
                    node.utility_proponent = random.randint(min_prop, max_prop)


    def prop_utilities_from_json(self, filename):
        with open(filename) as f:
            self.extra_data = json.load(f)
            for arg_id, util_value in self.extra_data["prop_utilities_frontier"].items():
                self.dict_tree[arg_id].utility_proponent = util_value
                if arg_id in self.extra_data["same_as"]:
                    for same_arg in self.extra_data["same_as"][arg_id]:
                        self.dict_tree[same_arg].utility_proponent = util_value


    def results_to_df(self):
        header = ["profile_id", "Node id", "Proponent utility", "Opponent utility", "Q_proponent_root", "Q_opponent_root"]
        output_df = pd.DataFrame(self.node_results, columns=header)
        #output_df.to_csv (r'table_results.csv', index = False, header=True)
        return output_df


    def predict(self, data_df, use_user_model=False):
        self.reset_results()
        for sample_id, sample in data_df.iterrows():
            # set opponent utilities
            self.reset_opponent_utilities()
            self.set_opponent_utilities(sample)
            if use_user_model:
                # leverage the model of the user for the inner chance nodes
                self.reset_user_model()
                self.set_user_model(sample)
            self.root.propagate_utility(policy='bimaximax')
            output_node = self.simulate_dialogue(sample_id, sample, use_user_model)
            # self.to_pdf(sample["id"])
            self.node_results.append([sample['id'], output_node.id, output_node.Q_proponent,
            output_node.Q_opponent, self.root.Q_proponent, self.root.Q_opponent])
        return self.results_to_df()


    def simulate_dialogue(self, sample_id, sample, user_model_on):
        tmp_node = self.root
        while not tmp_node.isLeaf():
            if tmp_node.is_decision:
                next_node = tmp_node.labelling
            else:
                next_node = self.get_user_choice(tmp_node, user_model_on=user_model_on)
            tmp_node = next_node
            # print(f"{sample_id} - {sample['id']} -> {next_node}")
        return tmp_node

    def get_user_choice(self, node, user_model_on, simulated=True, pick_first=False):
        """
        return user choice, if simulated look at the user model, otherwise ask to user
        """
        selected_node = None
        if simulated:
            if user_model_on:
                # for each child of the node extract id and user's preference based on the user model
                tmp_children_dict = {child.id: self.user_model[child.id] for child in node.children}
            else:
                # for each child of the node extract id and user's preference based on the backed-up utility
                tmp_children_dict = {child.id: child.Q_opponent for child in node.children}

            # sort according user's preference
            tmp_children_dict = dict(sorted(tmp_children_dict.items(), key=itemgetter(1, 0), reverse=True))

            # pick the node with highest preference
            if pick_first:
                selected_node = self.dict_tree[list(tmp_children_dict.keys())[0]]
            else:
                tmp_list = list(tmp_children_dict.items())
                best = tmp_list[0]
                output_list = [el[0] for el in tmp_list if el[1] == best[1]]
                random_node = random.choice(output_list)
                selected_node = self.dict_tree[random_node]
        else: # if simulated
            pass # TODO interactive mode
        return selected_node


    def set_opponent_utilities(self, sample, predicted=False):
        for id_arg in self.extra_data["frontier"]:
            self.dict_tree[id_arg].utility_opponent = int(sample[id_arg]) if type(sample[id_arg]) is str else sample[id_arg]

            if id_arg in self.extra_data["same_as"]:
                for same_arg in self.extra_data["same_as"][id_arg]:
                    self.dict_tree[same_arg].utility_opponent = int(sample[id_arg]) if type(sample[id_arg]) is str else sample[id_arg]
