"""CLI entry point: `python -m process_model [--grade G] [--tph T]`."""
import argparse
import time

from .optimizer import Optimizer, print_leaderboard


def main() -> None:
    ap = argparse.ArgumentParser(prog="process_model",
                                 description="Run the Cu plant TEA optimizer.")
    ap.add_argument("--grade", type=float, default=0.008,
                    help="Cu head grade as fraction (default: 0.008 = 0.8%%)")
    ap.add_argument("--tph", type=float, default=2000.0,
                    help="Feed throughput in tph (default: 2000)")
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    opt = Optimizer(feed_tph=args.tph, feed_grade=args.grade)
    res = opt.run(maxiter=args.maxiter, popsize=args.popsize,
                  workers=args.workers, disp=False)
    print_leaderboard(res)
    print(f"\n(elapsed: {time.time()-t0:.1f} s)")
    win = res.winner
    irr_s = f"{win.irr*100:.2f}%" if win.irr is not None else "n/a"
    print(f"NPV: ${win.best_npv/1e9:.2f}B")
    print(f"IRR: {irr_s}")


if __name__ == "__main__":
    main()
