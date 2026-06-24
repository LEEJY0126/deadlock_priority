"""Benchmark MST baseline vs a learned checkpoint on held-out instances."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.eval.benchmark import (make_eval_maps, make_instances, evaluate,
                                baseline_provider, print_report)
from src.priority.model import load_model, predict_field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="learned model checkpoint")
    ap.add_argument("--n_per_kind", type=int, default=10)
    ap.add_argument("--n_inst", type=int, default=4)
    ap.add_argument("--n_agents", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    maps = make_eval_maps(n_per_kind=args.n_per_kind)
    inst = make_instances(maps, n_agents=args.n_agents, n_inst=args.n_inst)

    print_report("MST baseline", evaluate(baseline_provider, inst))

    if args.ckpt:
        model = load_model(args.ckpt, device=args.device)
        provider = lambda g: predict_field(model, g, device=args.device)
        print_report(f"Learned ({os.path.basename(args.ckpt)})", evaluate(provider, inst))


if __name__ == "__main__":
    main()
