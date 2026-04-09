import os


def build_output_dir(root, dataset, experiment_name, phase):
    if not root:
        root = "outputs"
    if not dataset:
        dataset = "unknown_dataset"
    if not experiment_name:
        experiment_name = "unknown_experiment"
    if not phase:
        phase = "run"
    return os.path.join(root, str(dataset), str(experiment_name), str(phase))
