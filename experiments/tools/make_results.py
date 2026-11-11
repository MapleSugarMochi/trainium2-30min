"""Build the deliverable results.tsv from the runner's all_runs.tsv."""
import csv, sys
src, dst = sys.argv[1], sys.argv[2]
KEEP = {  # runs that established a change we kept (the chain to the final config)
    "manual_baseline_nc1", "sched_wd40_fix", "sched_b8c", "sched_fast", "sched_fa",
    "sched_fa_d12", "sched_fa_tb128k", "c_d12", "c_d12_s1", "k_qknorm_d12",
    "ck_d12", "ck_d12_s1", "final_mlr035", "final_mlr035_s1", "final_v1", "final_v2",
    "final_s1", "final_s2", "final_s3", "final_s4", "final_s5", "tp_bf16", "bf16_full", "bf16_d14", "bf16_tb192k", "f2_tb192k", "f2_tb256k", "f3_tb384k",
}
NOTE = {
    "f4_384_sc30": " [eval INVALID: softcap not stored in checkpoint, eval used 15; fixed on exp/softcap]",
    "f4_384_unet": " [OOM at device batch 8]",
    "f6_gc1": " [crash: Neuron cannot compile aten::_foreach_norm used by clip_grad_norm_]",
}
rows = [("commit", "val_bpb", "steps", "s_per_step", "status", "description")]
rows.append(("92240eb", "1.259076", "1000", "1.598", "keep",
             "baseline, lnc=1 x 8 procs, 1000-step cap hit at 1590 s (manual run)"))
with open(src) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        st = r["status"]
        if st != "crash":
            st = "keep" if r["id"] in KEEP else "discard"
        desc = f'{r["id"]}: {r["desc"]}'
        desc += NOTE.get(r["id"], "")
        if r["max_s"] != "1800":
            desc += f' [{r["max_s"]} s run]'
        rows.append((r["commit"], r["val_bpb"], r["steps"], r["s_per_step"], st, desc))
with open(dst, "w") as f:
    for row in rows:
        f.write("\t".join(row) + "\n")
print(len(rows) - 1, "runs written")
