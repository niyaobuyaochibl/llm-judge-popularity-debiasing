# Popularity-Aware Comparison

Baseline: `crossllm_deepseek`  
Candidate: `popaware_deepseek`  
Overlapping records: 20

## Overall
- `consistency_rate`: baseline 0.542, candidate 0.810, delta +0.268, improved in 17/20 records
- `alignment_to_ndcg`: baseline 0.600, candidate 0.550, delta -0.050, improved in 5/20 records
- `alignment_to_coverage`: baseline 0.600, candidate 0.750, delta +0.150, improved in 7/20 records
- `alignment_to_tail_exposure`: baseline 0.600, candidate 0.750, delta +0.150, improved in 7/20 records
- `alignment_to_gini`: baseline 0.600, candidate 0.850, delta +0.250, improved in 8/20 records
- `alignment_to_arp`: baseline 0.600, candidate 0.850, delta +0.250, improved in 8/20 records

## By Dimension
- `balanced`: delta consistency +0.082, delta coverage +0.200, delta tail_exposure +0.000
- `diversity`: delta consistency +0.454, delta coverage +0.100, delta tail_exposure +0.300
