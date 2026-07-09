"""Benchmark MST baseline vs a learned checkpoint on held-out instances."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.envs.simulator import ORACLES
from src.eval.benchmark import (make_eval_maps, make_instances, evaluate,
                                evaluate_embedding, evaluate_action, evaluate_elapsed,
                                baseline_provider, print_report, print_action_report,
                                print_inference_timing)
from src.priority.model import build_model, load_model, predict_field
from src.priority.model_embedding import EmbeddingPriorityModel, InferenceTimer
from src.priority.model_action import EmbeddingActionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="learned model checkpoint")
    ap.add_argument("--n_per_kind", type=int, default=10)
    ap.add_argument("--n_inst", type=int, default=4)
    ap.add_argument("--n_agents", type=int, default=8)
    ap.add_argument("--oracle", choices=ORACLES, default="paper",
                    help="PIBT resolution mode for the eval rollouts: paper "
                         "(right-hand rule + livelock), beta (legacy boost), or "
                         "goal-livelock (paper + goal-livelock retreat)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_elapsed_pibt", action="store_true",
                    help="skip the mapf-IR elapsed-time PIBT baseline (plain PIBT, "
                         "no field/oracle; reported by default for A/B)")
    ap.add_argument("--embedding", action="store_true",
                    help="evaluate the dynamic embedding model; with --ckpt loads "
                         "it, otherwise a fresh (untrained) model for a baseline")
    ap.add_argument("--action", action="store_true",
                    help="evaluate the dynamic action-map policy (no PIBT, direct "
                         "moves + collision termination); with --ckpt loads it, "
                         "otherwise a fresh (untrained) model for a baseline")
    args = ap.parse_args()

    maps = make_eval_maps(n_per_kind=args.n_per_kind)
    inst = make_instances(maps, n_agents=args.n_agents, n_inst=args.n_inst)

    if not args.no_elapsed_pibt:
        # plain PIBT with mapf-IR's dynamic priority -- ignores --oracle (it has no
        # field and no paper/beta/goal-livelock resolution); the canonical baseline.
        print_report("PIBT (mapf-IR elapsed-time)", evaluate_elapsed(inst))
    print_report("MST baseline", evaluate(baseline_provider, inst, oracle=args.oracle))

    if args.ckpt:
        model = load_model(args.ckpt, device=args.device)
        name = f"Learned ({os.path.basename(args.ckpt)})"
        if isinstance(model, EmbeddingActionModel):
            # direct action policy -> greedy rollouts (no PIBT); report collisions
            timer = InferenceTimer()
            print_action_report(name, evaluate_action(model, inst,
                                                      device=args.device, timer=timer))
            print_inference_timing(name, timer)
        elif isinstance(model, EmbeddingPriorityModel):
            # dynamic field -> per-instance field_fn rollouts; time enc/dec
            timer = InferenceTimer()
            print_report(name, evaluate_embedding(model, inst, oracle=args.oracle,
                                                  device=args.device, timer=timer))
            print_inference_timing(name, timer)
        else:
            provider = lambda g: predict_field(model, g, device=args.device)
            print_report(name, evaluate(provider, inst, oracle=args.oracle))
    elif args.action:
        torch.manual_seed(0)  # reproducible untrained baseline
        model = build_model("embedding_action").to(args.device)
        timer = InferenceTimer()
        print_action_report("Action policy (untrained)",
                            evaluate_action(model, inst, device=args.device, timer=timer))
        print_inference_timing("Action policy (untrained)", timer)
    elif args.embedding:
        torch.manual_seed(0)  # reproducible untrained baseline
        model = build_model("embedding").to(args.device)
        timer = InferenceTimer()
        print_report("Embedding (untrained)",
                     evaluate_embedding(model, inst, oracle=args.oracle,
                                        device=args.device, timer=timer))
        print_inference_timing("Embedding (untrained)", timer)


if __name__ == "__main__":
    main()
