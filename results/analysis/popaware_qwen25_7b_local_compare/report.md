# Popularity-Aware Comparison

Baseline: `crossllm_qwen25_7b_local`  
Candidate: `popaware_qwen25_7b_local`  
Overlapping records: 20

## Overall
- `consistency_rate`: baseline 0.262, candidate 0.550, delta +0.288, improved in 14/20 records
- `alignment_to_ndcg`: baseline 0.450, candidate 0.500, delta +0.050, improved in 7/20 records
- `alignment_to_coverage`: baseline 0.600, candidate 0.650, delta +0.050, improved in 7/20 records
- `alignment_to_tail_exposure`: baseline 0.600, candidate 0.600, delta +0.000, improved in 6/20 records
- `alignment_to_gini`: baseline 0.550, candidate 0.750, delta +0.200, improved in 9/20 records
- `alignment_to_arp`: baseline 0.550, candidate 0.700, delta +0.150, improved in 8/20 records

## By Dimension
- `balanced`: delta consistency +0.318, delta coverage +0.200, delta tail_exposure +0.100
- `diversity`: delta consistency +0.258, delta coverage -0.100, delta tail_exposure -0.100
