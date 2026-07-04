import sys
sys.path.insert(0, '.')
import _init_paths
from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset, trackerlist

dataset_name = 'otb_lang'
trackers = []
trackers += trackerlist('lantrack', 'lantrack_256_otb99lang_struct_onlyv2_token_reweight_concept_target_span_topk8_ep30', dataset_name, 30, 'otb_train_top8_ep30')
trackers += trackerlist('lantrack', 'lantrack_256_tnl2k_struct_onlyv2_token_reweight_concept_target_span_topk8_ep30', dataset_name, 30, 'tnl2k_top8_ep30')
trackers += trackerlist('lantrack', 'lantrack_256_tnllt_struct_onlyv2_token_reweight_concept_target_span_topk8', dataset_name, 30, 'tnllt_top8_ep30')
trackers += trackerlist('lantrack', 'lantrack_256_tnl2k_struct_onlyv2_token_reweight_concept_target_span_topk8_ep50', dataset_name, 30, 'tnllt_top8_ep50')
trackers += trackerlist('lantrack', 'lantrack_256_tnllt_tnl2k_otb99lang_struct_onlyv2_token_reweight_concept_target_span_topk8_ep30', dataset_name, 30, 'tnllt_mix_top8_ep30')
trackers += trackerlist('lantrack', 'lantrack_256_tnllt_tnl2k_otb99lang_struct_onlyv2_token_reweight_concept_target_span_topk8_ep30_test_cache_mixed_all_r16_lr3e5_e5', dataset_name, 30, 'tnllt_mix_top8_ep30_LORA')
dataset = get_dataset(dataset_name)
print_results(trackers, dataset, dataset_name, merge_results=True,
              plot_types=('success', 'norm_prec', 'prec'), force_evaluation=True)
