import json
d = json.load(open('/root/benchmarks/bench_specialization_results.json'))
for r in d['results']:
    bm = r['baseline_metrics']
    sm = r['submitted_metrics']
    print(f"{r['scenario']:35s} hit={str(r['specialization_hit']):5s} sp={r['speedup']:.4f}  mask:{bm['mask_complexity']}->{sm['mask_complexity']}  stride:{bm['stride_expr_count']}->{sm['stride_expr_count']}  ptr:{bm['pointer_expr_count']}->{sm['pointer_expr_count']}")
