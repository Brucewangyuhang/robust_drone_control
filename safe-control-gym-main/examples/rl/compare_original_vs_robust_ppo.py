'''Native-task disturbance check for original PPO vs robust path-following PPO.

The original safe-control-gym PPO checkpoints are evaluated in their native
quadrotor_3D strict time-indexed tracking task with an added impulse disturbance.
The robust checkpoint is evaluated in its own disturbance-aware path-following
completion task. This avoids migrating the original models into the new task.
'''

import argparse
import copy
import csv
import os
import sys
from functools import partial
from types import SimpleNamespace

import numpy as np
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import safe_control_gym  # noqa: F401  # Registers envs/controllers.
from evaluate_robust_path_follow_ppo import evaluate_magnitude, set_eval_disturbance
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.registration import get_config, make
from safe_control_gym.utils.utils import merge_dict, read_file


SCRIPT_DIR = os.path.dirname(__file__)


DEFAULT_ORIGINAL_PPO = os.path.join(
    SCRIPT_DIR,
    'models',
    'ppo',
    'ppo_model_quadrotor_3D_track.pt',
)
DEFAULT_ORIGINAL_SAFE_PPO = os.path.join(
    SCRIPT_DIR,
    'models',
    'safe_explorer_ppo',
    'safe_explorer_ppo_model_quadrotor_3D_track.pt',
)
DEFAULT_ROBUST_RUN_DIR = os.path.join(
    SCRIPT_DIR,
    'results',
    'ppo_quadrotor_3D_figure8_completion_v3_4400_recovery_penalty_seed2',
)
DEFAULT_ROBUST_CHECKPOINT = os.path.join(
    'checkpoints',
    'model_4552000.pt',
)
DEFAULT_NATIVE_TASK_CONFIG = os.path.join(
    SCRIPT_DIR,
    'config_overrides',
    'quadrotor_3D',
    'quadrotor_3D_track.yaml',
)
DEFAULT_PPO_CONFIG = os.path.join(
    SCRIPT_DIR,
    'config_overrides',
    'quadrotor_3D',
    'ppo_quadrotor_3D.yaml',
)
DEFAULT_SAFE_PPO_CONFIG = os.path.join(
    SCRIPT_DIR,
    'config_overrides',
    'quadrotor_3D',
    'safe_explorer_ppo_quadrotor_3D.yaml',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='./results/original_vs_robust_ppo_comparison')
    parser.add_argument('--original-ppo', type=str, default=DEFAULT_ORIGINAL_PPO)
    parser.add_argument('--original-safe-ppo', type=str, default=DEFAULT_ORIGINAL_SAFE_PPO)
    parser.add_argument('--robust-run-dir', type=str, default=DEFAULT_ROBUST_RUN_DIR)
    parser.add_argument('--robust-checkpoint', type=str, default=DEFAULT_ROBUST_CHECKPOINT,
                        help='Robust checkpoint path. Relative paths are resolved from --robust-run-dir.')
    parser.add_argument('--native-task-config', type=str, default=DEFAULT_NATIVE_TASK_CONFIG)
    parser.add_argument('--baseline-task-mode', type=str, default='native', choices=['native'],
                        help='Kept for compatibility; original models are always evaluated in their native task.')
    parser.add_argument('--ppo-config', type=str, default=DEFAULT_PPO_CONFIG)
    parser.add_argument('--safe-ppo-config', type=str, default=DEFAULT_SAFE_PPO_CONFIG)
    parser.add_argument('--magnitudes', nargs='+', type=float, default=[0.0, 0.3, 0.35, 0.4])
    parser.add_argument('--n-episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--step-offset-low', type=int, default=75)
    parser.add_argument('--step-offset-high', type=int, default=200)
    parser.add_argument('--duration', type=int, default=5)
    parser.add_argument('--decay-rate', type=float, default=0.85)
    parser.add_argument('--save-plot', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--plot-episode', type=int, default=0)
    parser.add_argument('--visualization-speed', type=float, default=1.0)
    return parser.parse_args()


def abs_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.abspath(path)


def robust_checkpoint_path(run_dir, checkpoint):
    if os.path.isabs(checkpoint):
        return checkpoint
    return os.path.join(run_dir, checkpoint)


def load_run_config(run_dir):
    config_path = os.path.join(run_dir, 'config.yaml')
    with open(config_path, 'r', encoding='UTF-8') as file:
        return yaml.safe_load(file)


def strict_baseline_config(algo, algo_override_path, task_override_path):
    '''Build a config matching the original model architecture and strict eval task.'''

    config = {
        'task': 'quadrotor',
        'algo': algo,
        'task_config': get_config('quadrotor'),
        'algo_config': get_config(algo),
        'use_gpu': False,
    }
    merge_dict(config, read_file(algo_override_path))
    merge_dict(config, read_file(task_override_path))

    # These are evaluation-only configs; avoid accidental training/pretraining behavior.
    config['algo_config']['training'] = False
    if algo == 'safe_explorer_ppo':
        config['algo_config']['pretraining'] = False
        config['algo_config']['pretrained'] = None
    return config


def eval_args_for_method(base_args, output_dir, seed_episodes=True):
    '''Small namespace with the fields expected by evaluate_magnitude().'''

    return SimpleNamespace(
        run_dir=output_dir,
        gui=base_args.gui,
        record=base_args.record,
        plot=base_args.plot,
        save_plot=base_args.save_plot,
        plot_episode=base_args.plot_episode,
        visualization_speed=base_args.visualization_speed,
        n_episodes=base_args.n_episodes,
        seed=base_args.seed,
        seed_episodes=seed_episodes,
        step_offset_low=base_args.step_offset_low,
        step_offset_high=base_args.step_offset_high,
        duration=base_args.duration,
        decay_rate=base_args.decay_rate,
    )


def danger_ratio(row, config):
    steps = float(config['task_config'].get('episode_len_sec', 5) * config['task_config'].get('ctrl_freq', 50))
    danger_steps = float(row.get('average_danger_zone_steps', np.nan))
    if not np.isfinite(danger_steps):
        return np.nan
    return danger_steps / max(1.0, steps)


def scalar(metrics, key):
    value = metrics.get(key, np.nan)
    if isinstance(value, np.ndarray):
        return float(np.nanmean(value))
    return float(value)


def info_series(info_episode, key, default=0.0):
    values = []
    for info in info_episode:
        values.append(info.get(key, default) if isinstance(info, dict) else default)
    return np.asarray(values, dtype=float)


def shade_disturbance(ax, t, disturbance_active):
    if disturbance_active.size == 0 or not np.any(disturbance_active > 0):
        return
    active = disturbance_active > 0
    start = None
    for idx, is_active in enumerate(active):
        if is_active and start is None:
            start = idx
        ended = start is not None and (not is_active or idx == len(active) - 1)
        if ended:
            end = idx if not is_active else idx + 1
            ax.axvspan(t[start], t[min(end - 1, len(t) - 1)], color='tab:red', alpha=0.15)
            start = None


def native_plot_output_path(args, checkpoint, magnitude):
    mag_label = f'{magnitude:.2f}'.replace('.', 'p').replace('-', 'm')
    label = os.path.splitext(os.path.basename(checkpoint))[0]
    out_dir = os.path.join(args.run_dir, 'evals', 'plots')
    return os.path.join(out_dir, f'native_trajectory_{label}_mag_{mag_label}.png')


def target_score(row, config):
    '''Task-level score for ranking policies; return is intentionally not used.'''

    threshold = float(config['task_config'].get('path_error_threshold', 0.35))
    path_error = float(row['average_path_error'])
    peak_error = float(row['average_peak_path_error'])
    progress = float(row['average_final_progress'])
    coverage = float(row['average_path_coverage'])
    completion = float(row['completion_rate'])
    danger = danger_ratio(row, config)
    peak_after = row.get('average_peak_deviation_after_disturbance', np.nan)
    peak_after_penalty = 0.0 if not np.isfinite(peak_after) else max(0.0, float(peak_after) / max(threshold, 1e-6) - 1.0)

    return (
        4.0 * completion
        + 3.0 * coverage
        + 2.0 * progress
        - 2.0 * min(path_error / max(threshold, 1e-6), 4.0)
        - 1.0 * min(peak_error / max(threshold, 1e-6), 4.0)
        - 2.0 * min(danger, 1.0)
        - 0.5 * min(peak_after_penalty, 4.0)
    )


def target_quality(row, config):
    '''Readable target-task quality independent of cross-method reward scale.'''

    threshold = float(config['task_config'].get('path_error_threshold', 0.35))
    path_error = float(row['average_path_error'])
    progress = float(row['average_final_progress'])
    coverage = float(row['average_path_coverage'])
    completion = float(row['completion_rate'])
    danger = danger_ratio(row, config)

    if completion >= 0.8 and progress >= 0.9 and coverage >= 0.85 and path_error <= threshold and danger <= 0.10:
        quality = 'good'
    elif progress >= 0.5 and coverage >= 0.5 and path_error <= 2.0 * threshold and danger <= 0.35:
        quality = 'partial'
    else:
        quality = 'poor'
    note = (
        f'completion={completion:.2f}, progress={progress:.2f}, coverage={coverage:.2f}, '
        f'path_error={path_error:.3f}, danger_ratio={danger:.2f}'
    )
    return quality, note


def plot_original_native_evaluation(results, x_goal, row, config, checkpoint_path, magnitude, args):
    '''Original-project-style trajectory plot for native strict-tracking baselines.'''

    import matplotlib.pyplot as plt

    n_episodes = len(results['state'])
    episode_idx = min(max(0, args.plot_episode), n_episodes - 1)
    states = np.asarray(results['state'][episode_idx])
    rewards = np.asarray(results['reward'][episode_idx], dtype=float)
    info = results['info'][episode_idx]
    ctrl_freq = float(config['task_config'].get('ctrl_freq', 50))

    pos = states[:, [0, 2, 4]]
    ref_pos = np.asarray(x_goal)[:, [0, 2, 4]]
    t_state = np.arange(pos.shape[0]) / ctrl_freq
    t_reward = np.arange(rewards.shape[0]) / ctrl_freq
    t_info = np.arange(len(info)) / ctrl_freq
    disturbance_active = info_series(info, 'disturbance_active', default=0.0)

    fig = plt.figure(figsize=(13, 9))
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax3d.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2], 'g--', label='reference trajectory')
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'r-', label='policy trajectory')
    ax3d.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color='tab:blue', s=35, label='start')
    ax3d.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color='tab:red', s=35, label='end')
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    ax3d.legend(loc='best')

    ax_xz = fig.add_subplot(2, 2, 2)
    ax_xz.plot(ref_pos[:, 0], ref_pos[:, 2], 'g--', label='reference trajectory')
    ax_xz.plot(pos[:, 0], pos[:, 2], 'r-', label='policy trajectory')
    ax_xz.scatter(pos[0, 0], pos[0, 2], color='tab:blue', s=35, label='start')
    ax_xz.scatter(pos[-1, 0], pos[-1, 2], color='tab:red', s=35, label='end')
    ax_xz.set_xlabel('x')
    ax_xz.set_ylabel('z')
    ax_xz.set_title('x-z projection')
    ax_xz.axis('equal')
    ax_xz.legend(loc='best')

    ax_time = fig.add_subplot(2, 2, 3)
    n_ref = min(pos.shape[0], ref_pos.shape[0])
    t_ref = np.arange(n_ref) / ctrl_freq
    ax_time.plot(t_state, pos[:, 0], color='tab:blue', label='x')
    ax_time.plot(t_ref, ref_pos[:n_ref, 0], color='tab:blue', linestyle='--', label='x ref')
    ax_time.plot(t_state, pos[:, 2], color='tab:green', label='z')
    ax_time.plot(t_ref, ref_pos[:n_ref, 2], color='tab:green', linestyle='--', label='z ref')
    shade_disturbance(ax_time, t_info, disturbance_active)
    ax_time.set_xlabel('time (s)')
    ax_time.set_ylabel('position')
    ax_time.set_title('native time-indexed tracking')
    ax_time.legend(loc='best')

    ax_reward = fig.add_subplot(2, 2, 4)
    if rewards.size:
        ax_reward.plot(t_reward, rewards, color='tab:orange', label='reward')
    shade_disturbance(ax_reward, t_info, disturbance_active)
    ax_reward.set_xlabel('time (s)')
    ax_reward.set_ylabel('reward')
    ax_reward.set_title('reward and disturbance window')
    ax_reward.legend(loc='best')

    fig.suptitle(
        f'{os.path.basename(checkpoint_path)} | native strict tracking | magnitude={magnitude:.2f} | '
        f'return={row["average_return"]:.3f}, rmse={row["average_rmse"]:.3f}, failure_rate={row["failure_rate"]:.2f}'
    )
    fig.tight_layout()

    saved_path = None
    if args.save_plot:
        saved_path = native_plot_output_path(args, checkpoint_path, magnitude)
        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
        fig.savefig(saved_path, dpi=180)
        print(f'Saved native trajectory plot to {saved_path}')
    if args.plot:
        plt.show()
    else:
        plt.close(fig)
    return saved_path


def evaluate_original_native_magnitude(config, checkpoint_path, magnitude, args):
    '''Evaluate an original checkpoint in its native task, only injecting impulse disturbance.'''

    task_config = set_eval_disturbance(config['task_config'], magnitude, args)
    algo_config = copy.deepcopy(config['algo_config'])
    algo_config.pop('training', None)

    env_func = partial(make, config['task'], **task_config)
    env = env_func(
        gui=args.gui,
        record=args.record,
        output_dir=args.run_dir,
    )
    ctrl = make(
        config['algo'],
        env_func,
        training=False,
        output_dir=os.path.join(args.run_dir, 'eval_temp'),
        **algo_config,
    )
    ctrl.load(checkpoint_path)

    experiment = BaseExperiment(env, ctrl)
    results, metrics = experiment.run_evaluation(
        n_episodes=args.n_episodes,
        verbose=False,
        visualization_time_multiplier=args.visualization_speed,
    )
    x_goal = np.array(env.X_GOAL, copy=True)
    experiment.close()

    row = {
        'checkpoint': os.path.basename(checkpoint_path),
        'magnitude': magnitude,
        'average_length': scalar(metrics, 'average_length'),
        'average_return': scalar(metrics, 'average_return'),
        'average_rmse': scalar(metrics, 'average_rmse'),
        'failure_rate': scalar(metrics, 'failure_rate'),
        'average_constraint_violation': scalar(metrics, 'average_constraint_violation'),
        'average_path_error': np.nan,
        'average_peak_path_error': np.nan,
        'average_final_progress': np.nan,
        'average_path_coverage': np.nan,
        'completion_rate': np.nan,
        'average_danger_zone_steps': np.nan,
        'average_recovery_time': np.nan,
        'average_peak_deviation_after_disturbance': np.nan,
        'quality': 'native_check',
        'quality_note': 'native strict-tracking task; path-completion metrics intentionally not used',
    }
    if args.plot or args.save_plot:
        plot_path = plot_original_native_evaluation(results, x_goal, row, config, checkpoint_path, magnitude, args)
        if plot_path is not None:
            row['plot_path'] = plot_path
    return row


def add_comparison_fields(row, method, config):
    row = copy.deepcopy(row)
    row['method'] = method['name']
    row['method_family'] = method['family']
    row['eval_protocol'] = method['protocol']
    row['reference_mode'] = config['task_config'].get('tracking_reference_mode', 'unknown')
    row['model_path'] = method['checkpoint']
    row['danger_ratio'] = danger_ratio(row, config)
    if method['family'] == 'ours':
        row['target_score'] = target_score(row, config)
        row['target_quality'], row['target_note'] = target_quality(row, config)
    else:
        row['target_score'] = np.nan
        row['target_quality'] = 'native_check'
        row['target_note'] = (
            f'native_return={row["average_return"]:.3f}, rmse={row["average_rmse"]:.3f}, '
            f'length={row.get("average_length", np.nan):.1f}, failure_rate={row["failure_rate"]:.2f}'
        )
    return row


def format_float(value, digits=3):
    if value is None:
        return 'nan'
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return 'nan'
    return f'{value:.{digits}f}'


def nominal_by_method(rows):
    nominal = {}
    for row in rows:
        if abs(float(row['magnitude'])) < 1e-9:
            nominal[row['method']] = row
    return nominal


def delta(row, nominal, key):
    if nominal is None:
        return np.nan
    return float(row[key]) - float(nominal[key])


def pct_delta(row, nominal, key):
    if nominal is None:
        return np.nan
    base = float(nominal[key])
    if abs(base) < 1e-9:
        return np.nan
    return 100.0 * (float(row[key]) - base) / abs(base)


def write_conclusion(rows, out_path):
    lines = []
    lines.append('# Native Original Policies vs Robust Path-Following PPO')
    lines.append('')
    lines.append('Comparison setup:')
    lines.append('- Original PPO and original safe-explorer PPO are evaluated in their native `quadrotor_3D_track.yaml` strict time-indexed tracking task.')
    lines.append('- Robust PPO is evaluated in its own disturbance-aware path-following figure-8 completion task.')
    lines.append('- The impulse disturbance magnitude, duration, decay, seed protocol, and offset window are shared.')
    lines.append('- This is not a task-migration benchmark; original models are not moved into the new robust path-following environment.')
    lines.append('- For original models, read native tracking degradation metrics such as return, RMSE, and failure rate. For robust PPO, read path completion/recovery metrics.')
    lines.append('')

    nominal = nominal_by_method(rows)
    original_rows = [row for row in rows if row['method_family'] == 'original_baseline']
    robust_rows = [row for row in rows if row['method'] == 'robust_path_following_ppo']

    lines.append('## Original Native-Task Disturbance Sensitivity')
    lines.append('')
    lines.append('| method | magnitude | length | return | return_delta_% | rmse | rmse_delta_% | failure_rate | failure_delta |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for row in original_rows:
        base = nominal.get(row['method'])
        lines.append(
            '| {method} | {mag} | {length} | {ret} | {ret_delta} | {rmse} | {rmse_delta} | {failure} | {failure_delta} |'.format(
                method=row['method'],
                mag=format_float(row['magnitude'], 2),
                length=format_float(row.get('average_length', np.nan), 1),
                ret=format_float(row['average_return']),
                ret_delta=format_float(pct_delta(row, base, 'average_return')),
                rmse=format_float(row['average_rmse']),
                rmse_delta=format_float(pct_delta(row, base, 'average_rmse')),
                failure=format_float(row['failure_rate'], 2),
                failure_delta=format_float(delta(row, base, 'failure_rate'), 2),
            )
        )
    lines.append('')

    lines.append('## Robust Path-Following Recovery')
    lines.append('')
    lines.append('| magnitude | target_quality | completion | progress | coverage | path_error | peak_error | danger_ratio | peak_after_disturbance |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|')
    for row in robust_rows:
        lines.append(
            '| {mag} | {quality} | {completion} | {progress} | {coverage} | {path_error} | {peak} | {danger} | {peak_after} |'.format(
                mag=format_float(row['magnitude'], 2),
                quality=row['target_quality'],
                completion=format_float(row['completion_rate'], 2),
                progress=format_float(row['average_final_progress'], 2),
                coverage=format_float(row['average_path_coverage'], 2),
                path_error=format_float(row['average_path_error']),
                peak=format_float(row['average_peak_path_error']),
                danger=format_float(row['danger_ratio'], 2),
                peak_after=format_float(row['average_peak_deviation_after_disturbance']),
            )
        )
    lines.append('')
    lines.append('Interpretation guide:')
    lines.append('- For original native policies, a large return drop, RMSE increase, or failure-rate increase under impulse indicates weak disturbance robustness in the original strict-tracking formulation.')
    lines.append('- For robust PPO, high completion/coverage with low path error and danger ratio indicates post-disturbance recovery and continuation.')
    lines.append('- Do not compare return directly between original and robust PPO, because they run different task formulations and reward definitions.')

    with open(out_path, 'w', encoding='UTF-8') as file:
        file.write('\n'.join(lines) + '\n')


def print_row(row):
    if row['method_family'] == 'original_baseline':
        print(
            '{method:28s} mag={magnitude:.2f} protocol={eval_protocol} '
            'length={average_length:.1f} return={average_return:.3f} rmse={average_rmse:.3f} '
            'failure={failure_rate:.2f}'.format(**row)
        )
    else:
        print(
            '{method:28s} mag={magnitude:.2f} protocol={eval_protocol} target={target_quality:7s} '
            'completion={completion_rate:.2f} progress={average_final_progress:.3f} '
            'coverage={average_path_coverage:.3f} path_error={average_path_error:.3f} '
            'peak={average_peak_path_error:.3f} danger_ratio={danger_ratio:.2f}'.format(**row)
        )


def main():
    args = parse_args()
    output_dir = abs_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    original_ppo = abs_path(args.original_ppo)
    original_safe_ppo = abs_path(args.original_safe_ppo)
    robust_run_dir = abs_path(args.robust_run_dir)
    robust_ckpt = robust_checkpoint_path(robust_run_dir, args.robust_checkpoint)

    for path in [original_ppo, original_safe_ppo, robust_ckpt]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Model checkpoint not found: {path}')

    baseline_task_config = args.native_task_config

    strict_ppo_config = strict_baseline_config(
        'ppo',
        abs_path(args.ppo_config),
        abs_path(baseline_task_config),
    )
    strict_safe_config = strict_baseline_config(
        'safe_explorer_ppo',
        abs_path(args.safe_ppo_config),
        abs_path(baseline_task_config),
    )
    robust_config = load_run_config(robust_run_dir)
    robust_config['algo_config']['training'] = False

    methods = [
        {
            'name': 'original_strict_ppo',
            'family': 'original_baseline',
            'protocol': 'native_strict_tracking',
            'checkpoint': original_ppo,
            'config': strict_ppo_config,
        },
        {
            'name': 'original_safe_explorer_ppo',
            'family': 'original_baseline',
            'protocol': 'native_strict_tracking',
            'checkpoint': original_safe_ppo,
            'config': strict_safe_config,
        },
        {
            'name': 'robust_path_following_ppo',
            'family': 'ours',
            'protocol': 'robust_path_following_completion',
            'checkpoint': robust_ckpt,
            'config': robust_config,
        },
    ]

    print('Comparison output directory:', output_dir)
    print('Baseline task mode:', args.baseline_task_mode)
    print('Baseline task config:', abs_path(baseline_task_config))
    print('Magnitudes:', ', '.join(f'{mag:.2f}' for mag in args.magnitudes))
    print('Episodes per method/magnitude:', args.n_episodes)

    rows = []
    for method in methods:
        print(f'\n=== {method["name"]} ===')
        method_args = eval_args_for_method(args, output_dir, seed_episodes=method['family'] != 'original_baseline')
        for magnitude in args.magnitudes:
            if method['family'] == 'original_baseline':
                row = evaluate_original_native_magnitude(method['config'], method['checkpoint'], magnitude, method_args)
            else:
                row = evaluate_magnitude(method['config'], method['checkpoint'], magnitude, method_args)
            row = add_comparison_fields(row, method, method['config'])
            rows.append(row)
            print_row(row)

    # Keep the most useful columns first, then preserve all evaluator fields.
    preferred = [
        'method',
        'method_family',
        'eval_protocol',
        'reference_mode',
        'magnitude',
        'target_score',
        'target_quality',
        'target_note',
        'average_length',
        'completion_rate',
        'average_final_progress',
        'average_path_coverage',
        'average_path_error',
        'average_peak_path_error',
        'average_danger_zone_steps',
        'danger_ratio',
        'average_recovery_time',
        'average_peak_deviation_after_disturbance',
        'failure_rate',
        'average_constraint_violation',
        'average_return',
        'average_rmse',
        'quality',
        'quality_note',
        'checkpoint',
        'model_path',
        'plot_path',
    ]
    fieldnames = []
    for key in preferred:
        if any(key in row for row in rows):
            fieldnames.append(key)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    csv_path = os.path.join(output_dir, 'comparison_original_vs_robust.csv')
    with open(csv_path, 'w', newline='', encoding='UTF-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    conclusion_path = os.path.join(output_dir, 'comparison_conclusion.md')
    write_conclusion(rows, conclusion_path)

    print(f'\nSaved CSV: {csv_path}')
    print(f'Saved conclusion: {conclusion_path}')


if __name__ == '__main__':
    main()
