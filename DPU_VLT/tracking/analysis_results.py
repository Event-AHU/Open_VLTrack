import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist


trackers = []
dataset_name = 'tnllt' # lasot_extension_subset


# for TNLLT EVA baseline epoch45-50
trackers.extend(trackerlist(name='lantrack', parameter_name='lantrack_256_baseline', dataset_name=dataset_name,
                        display_name='baseline_tnllt_45',run_ids=45))
# trackers.extend(trackerlist(name='lantrack', parameter_name='lantrack_256_baseline', dataset_name=dataset_name,
#                         display_name='baseline_tnllt_46',run_ids=46))
# trackers.extend(trackerlist(name='lantrack', parameter_name='lantrack_256_baseline', dataset_name=dataset_name,
#                         display_name='baseline_tnllt_47',run_ids=47))
# trackers.extend(trackerlist(name='lantrack', parameter_name='lantrack_256_baseline', dataset_name=dataset_name,
#                         display_name='baseline_tnllt_48',run_ids=48))
# trackers.extend(trackerlist(name='lantrack', parameter_name='lantrack_256_baseline', dataset_name=dataset_name,
#                         display_name='baseline_tnllt_49',run_ids=49))

# for LASOT EVA
# trackers.extend(trackerlist(name='lantrack', parameter_name='lantrack_256_all_cac_langproto_N_2_landa_0.5_text_blip', dataset_name=dataset_name,
#                         display_name='lanTrack1',run_ids=45))
# trackers.extend(trackerlist(name='lantrack', parameter_name='lantrack_256_all_cac_mvp_N_2_landa_0.5', dataset_name=dataset_name,
#                         display_name='lanTrack2',run_ids=45))                        
# trackers.extend(trackerlist(name='lantrack', parameter_name='dutrack_raw', dataset_name=dataset_name,
#                         display_name='lanTrack3',run_ids=45))  
# For VOT evaluate
dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'),force_evaluation=True)

# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))

