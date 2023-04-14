import os
import re
import numpy as np

def metrics_hit(pred_dict, topK=10):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of tail_entity ids ordered by largest to smallest scores
    """
    # Compute metrics
    precisions, recalls, ndcgs, hits = [], [], [], []
    source_list = list(pred_dict.keys())
    for source_id in source_list:
        hit_count = 0
        for target, pred_list in pred_dict[source_id].items():
            if target in pred_list[0][:topK]:
                hit_count += 1
        hit = 1 if hit_count > 0 else 0
        hits.append(hit)
    avg_hit = np.mean(hits)
    return avg_hit

def get_pred_dict(datapath):
    pred_dict = {}
    with open(datapath, 'r')as f:
        lines = f.read()
        lines = lines.split('#####################\n')
        for line in lines[:-1]:
            line_ = re.split('Reward:[0-9]+\n', line)
            compound, disease, _ = re.split('\t|\n', line_[0])
            if compound in pred_dict.keys():
                pred_dict[compound].update({disease:[]})
            else:
                pred_dict[compound] = {disease: []}

            pred_list = line_[1].split('\n___\n')[:-1]
            pred_disease_list, pred_score_list = [], []
            for pred in pred_list:
                pred_disease = (pred.split('\n')[0]).rsplit('\t',1)[-1]
                pred_score = float(pred.rsplit('\n', 1)[-1])
                pred_disease_list.append(pred_disease)
                pred_score_list.append(pred_score)
            sort_idxs = np.argsort(pred_score_list)[::-1]
            pred_disease_list = np.array(pred_disease_list)[sort_idxs]
            pred_score_list = np.array(pred_score_list)[sort_idxs]
            pred_dict[compound][disease].extend([pred_disease_list,pred_score_list])
    return pred_dict


if __name__ == '__main__':
    datapath = '/mnt/home/ranting/software/PoLo-main/output/Hetionet/19Oct22_112421_TEST_p3_r3.0_e32_h32_L2_l0.0006_o0/test_beam/paths_CtD'
    # datapath = '/mnt/home/ranting/software/PoLo-main/output/Hetionet-newtrain/02Nov22_183033_TEST_p3_r3.0_e32_h32_L2_l0.0006_o0/test_beam/paths_CtD'
    pred_dict = get_pred_dict(datapath=datapath)
    hit_score1 = metrics_hit(pred_dict, topK=1)
    hit_score3 = metrics_hit(pred_dict, topK=3)
    hit_score10 = metrics_hit(pred_dict, topK=10)
    print(f"hit@1={hit_score1:.3f}, hit@3={hit_score3:.3f}, hit@10={hit_score10:.3f}")


