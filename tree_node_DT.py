from operator import attrgetter
import random

class TreeNode:
  """
  Class that defines a node of a decision tree
  """
  delta = 1.0

  def __init__(self, id, text):
    self.labelling = None
    self.text = text
    self.height = None
    self.id = id
    self.children = []
    self.utility_proponent = -1
    self.utility_opponent = -1
    self.Q_proponent = -1
    self.Q_opponent = -1
    self.is_decision = False # if it is a chance or decision node

  def set_utility_proponent(self, value):
    self.utility_proponent = value

  def add_child(self, node):
    assert isinstance(node, TreeNode)
    self.children.append(node)

  def set_children(self, children):
    self.children = children

  def set_utility_opponent(self, utility_opponent):
    self.utility_opponent = utility_opponent

  def compute_chance_decision(self, is_decision_node):
    self.is_decision = is_decision_node
    child_type = False if is_decision_node is True else True
    for child in self.children:
      child.compute_chance_decision(child_type)

  def __str__(self):
    node_type = "Decision" if self.is_decision is True else "Chance"
    base_str = f"{node_type} node {self.id} - {self.height}:\n{self.text}\n[{self.Q_proponent}, {self.Q_opponent}]"
    #for child in self.children:
    #  base_str += f"\n\t{child.id}: {child.text}"
    return base_str

  def isLeaf(self):
    return True if len(self.children) == 0 else False

  def AMax(self):
    """
    torna i figli del nodo con maxima utilita
    :return:
    """
    output_list = []
    tmp_list = self.children

    if self.is_decision:
      key = attrgetter('Q_proponent', 'id')
    else:
      key = attrgetter('Q_opponent', 'id')

    tmp_list.sort(key=key, reverse=False)
    max_util_node = tmp_list[-1]
    output_list.append(max_util_node)
    for i in range(0, len(tmp_list) - 1):
      if self.is_decision:
        if max_util_node.Q_proponent == tmp_list[i].Q_proponent:
          output_list.append(tmp_list[i])
      else:
        if max_util_node.Q_opponent == tmp_list[i].Q_opponent:
          output_list.append(tmp_list[i])
    return output_list

  def print_post_order(self):
    for n in self.children:
        n.print_post_order()
    print(self)

  def propagate_utility(self, policy):
    for child in self.children:
      child.propagate_utility(policy=policy)
    if policy == 'bimaximax':
      if self.isLeaf(): # the node is a leaf
        self.Q_proponent = self.utility_proponent
        self.Q_opponent = self.utility_opponent
      else:
        max_utility_child = self.choose_child()
        self.Q_proponent = self.delta*max_utility_child.Q_proponent
        self.Q_opponent = self.delta*max_utility_child.Q_opponent
        if self.is_decision:
          self.labelling = max_utility_child
        #if self.id == '1':
          #pdb.set_trace()
    else:
      print(f"Policy {policy} not implemented.")

  def choose_child(self, pick_first=False):
    """
    torna un figlio a casa del nodo con massima/minima utilita
    :return:
    """
    output_list = self.AMax()

    if pick_first:
      return output_list[0]
    else:
      return random.choice(output_list)

def load_samples(filename):
  samples = np.genfromtxt(filename, delimiter=',', names=True, dtype=None)
  return samples
