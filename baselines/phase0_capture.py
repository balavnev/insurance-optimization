import json, time
import torch
from offer_opt import metrics as M
from offer_opt.device import get_device

results = {}
for device_name, device in [("cpu", torch.device("cpu")), ("mps", get_device(prefer_gpu=True))]:
    results[device_name] = {}
    for case, kwargs in [("low", {}), ("med", {}), ("hard", dict(max_iters=400, repair_every=20))]:
        t0 = time.perf_counter()
        r = M.benchmark(case, device, n_reps=1, **kwargs)
        r["wallclock_capture_s"] = time.perf_counter() - t0
        results[device_name][case] = r
        print(device_name, case, "median_time=", r["median_time"], "total_ev=", r["total_ev"],
              "converged=", r["converged"], "iterations=", r["iterations"], "verifier_ok=", r["verifier_ok"])

with open("baselines/phase0_baseline.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("wrote baselines/phase0_baseline.json")
