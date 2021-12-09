import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict


def plot_mae(results, variance, title):
  """Printing the mean squared error"""
  fig = plt.figure(figsize=(15, 10))
  ax = plt.subplot(111)
  for method_name in results.keys():
    plt.errorbar(variance, results[method_name]['mae_cv'], yerr=results[method_name]['std_mae_cv'], label=method_name)
  plt.title(title)
  ax.legend()
  #plt.show()
  plt.savefig(f'{title}.pdf')


def build_synt_data_from_profiles(frontier_argument_ids_, profiles_, id_dem_atts, variance=1):
  """Build and save synthetic data from users profiles.

  Users profiles are repeated many times and added with some gaussian noise.
  """
  # type: (ndarray, ndarray, int) -> (ndarray, ndarray, List[str])

  col_name_idx_map = OrderedDict({col_name: i for i, col_name in enumerate(profiles_.dtype.names)})

  # Select data columns
  tmp_data: List[ndarray] = []
  for col_name, _ in col_name_idx_map.items():
    tmp_data.append(profiles_[col_name][:, np.newaxis])
  data = np.hstack(tmp_data)

  # repeat data
  repetitions = {0: 40, 1: 200, 2: 200, 3: 200, 4: 25, 5: 133}
  #repetitions = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 3: 1}
  tmp_data: List[ndarray] = []
  for profile_id in np.unique(profiles_['id']):
    samples_id = np.where(data[:, 0] == profile_id)[0]
    tmp_data.append(np.tile(data[samples_id], (2*repetitions[profile_id],1)))

  data = np.vstack(tmp_data)

  # Add Gaussian noise
  gaussian_noise = np.rint(np.random.normal(0, variance, data.shape))
  gaussian_noise[:, 0] = 0 # no noise for the id column
  data += gaussian_noise.astype(np.int32)
  """
  # Old code for adding noise only on certain columns
  noise_age_school = np.rint(np.random.normal(0, variance, (len(data),2)))
  noise_sex = np.rint(np.random.normal(0, variance, (len(data),1)))
  noise_meat_sporty = np.rint(np.random.normal(0, variance, (len(data),2)))
  noise_args = np.rint(np.random.normal(0, variance, (len(data),len(state_names_) - 6)))

  # no noise for the id column
  noise = np.hstack((np.zeros((len(data),1)), noise_age_school, noise_sex, noise_meat_sporty, noise_args))
  """
  # clip values outside the allowed variable range
  data[:, [1, 2]] = np.clip(data[:, [1, 2]], 0, 4) # age, school
  data[:, [3]] = np.clip(data[:, [3]], 0, 1) # sex
  data[:, 4:] = np.clip(data[:, 4:], 0, 10) # all the rest

  # shuffle
  np.random.shuffle(data)

  # Select columns/features for evidence
  evidence_idx: List[int] = [col_id for col_name, col_id in col_name_idx_map.items() if
                             (col_name in list(map(str, frontier_argument_ids_.tolist())) or
                              col_id < id_dem_atts)]

  evidence_names: List[str] = [col_name for col_name, col_id in col_name_idx_map.items() if
                             (col_name in list(map(str, frontier_argument_ids_.tolist())) or
                              col_id < id_dem_atts)]

  data_df = pd.DataFrame(data, columns=list(col_name_idx_map.keys()))
  return data[:, evidence_idx], data_df, evidence_names
