import warnings


def main() -> None:
    import optuna
    from sklearn.exceptions import ConvergenceWarning

    from .config import parse_args
    from .log import setup_logging
    from .pipeline import run

    args = parse_args()
    setup_logging(-1 if args.quiet else args.verbose)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run(args)
