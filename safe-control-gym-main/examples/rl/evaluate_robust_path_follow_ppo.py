'''Evaluate a full-retrained robust path-following PPO policy.

Runs the same policy across a sweep of impulse magnitudes and reports the
path-following/recovery metrics added by the robust task definition.
'''

import argparse
import copy
import csv
import os
import sys
from functools import partial

import numpy as np
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.registration import make


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, default='./results/ppo_quadrotor_3D_robust_path_follow')
    parser.add_argument('--model', type=str, default='best', choices=['best', 'latest'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Specific checkpoint .pt file to test. Relative paths are resolved from run_dir.')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Optional CSV output path. Defaults to run_dir/evals/robust_eval_<checkpoint>.csv.')
    parser.add_argument('--gui', action='store_true',
                        help='Open PyBullet GUI so you can watch the checkpoint fly during evaluation.')
    parser.add_argument('--record', action='store_true',
                        help='Record a PyBullet video while evaluating. With GUI this writes an mp4 under run_dir/videos.')
    parser.add_argument('--plot', action='store_true',
                        help='Show trajectory/reference plots after evaluation, like the original rl_experiment.py.')
    parser.add_argument('--save_plot', action='store_true',
                        help='Save trajectory/reference plots under run_dir/evals/plots.')
    parser.add_argument('--plot_episode', type=int, default=0,
                        help='Episode index to plot from each evaluated magnitude.')
    parser.add_argument('--visualization_speed', type=float, default=1.0,
                        help='GUI playback speed multiplier. 1.0 is real time, 2.0 is twice as fast.')
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--magnitudes', nargs='+', type=float, default=[0.0, 0.3, 0.5, 0.8, 1.0])
    parser.add_argument('--step_offset_low', type=int, default=25)
    parser.add_argument('--step_offset_high', type=int, default=300)
    parser.add_argument('--duration', type=int, default=5)
    parser.add_argument('--decay_rate', type=float, default=0.85)
    return parser.parse_args()


def load_run_config(run_dir):
    config_path = os.path.join(run_dir, 'config.yaml')
    with open(config_path, 'r', encoding='UTF-8') as file:
        return yaml.safe_load(file)


def model_path(run_dir, model):
    filename = 'model_best.pt' if model == 'best' else 'model_latest.pt'
    path = os.path.join(run_dir, filename)
    if model == 'best' and not os.path.exists(path):
        path = os.path.join(run_dir, 'model_latest.pt')
    return path


def checkpoint_path(args):
    if args.checkpoint is None:
        return model_path(args.run_dir, args.model)
    if os.path.isabs(args.checkpoint):
        return args.checkpoint
    return os.path.join(args.run_dir, args.checkpoint)


def checkpoint_label(path):
    label = os.path.splitext(os.path.basename(path))[0]
    return ''.join(char if char.isalnum() or char in {'_', '-'} else '_' for char in label)


def output_path(args, checkpoint):
    if args.output_csv is not None:
        return args.output_csv if os.path.isabs(args.output_csv) else os.path.join(args.run_dir, args.output_csv)
    out_dir = os.path.join(args.run_dir, 'evals')
    return os.path.join(out_dir, f'robust_eval_{checkpoint_label(checkpoint)}.csv')


def plot_output_path(args, checkpoint, magnitude):
    mag_label = f'{magnitude:.2f}'.replace('.', 'p').replace('-', 'm')
    out_dir = os.path.join(args.run_dir, 'evals', 'plots')
    return os.path.join(out_dir, f'trajectory_{checkpoint_label(checkpoint)}_mag_{mag_label}.png')


def set_eval_disturbance(task_config, magnitude, args):
    task_config = copy.deepcopy(task_config)
    if magnitude <= 0:
        task_config['disturbances'] = None
        return task_config

    task_config['disturbances'] = {
        'dynamics': [{
            'disturbance_func': 'impulse',
            'apply_probability': 1.0,
            'magnitude': magnitude,
            'random_direction': True,
            'step_offset_range': [args.step_offset_low, args.step_offset_high],
            'duration': args.duration,
            'decay_rate': args.decay_rate,
        }]
    }
    return task_config


def scalar(metrics, key):
    value = metrics.get(key, np.nan)
    if isinstance(value, np.ndarray):
        return float(np.nanmean(value))
    return float(value)


def quality_assessment(row, task_config):
    '''Human-readable quality label for quick checkpoint triage.'''
    path_threshold = float(task_config.get('path_error_threshold', 0.35))
    episode_steps = float(task_config.get('episode_len_sec', 5) * task_config.get('ctrl_freq', 50))
    path_error = row['average_path_error']
    progress = row['average_final_progress']
    coverage = row.get('average_path_coverage', progress)
    completion_rate = row.get('completion_rate', 0.0)
    danger_ratio = row['average_danger_zone_steps'] / max(1.0, episode_steps)
    failure_rate = row['failure_rate']

    if failure_rate > 0.50:
        label = 'poor'
    elif completion_rate >= 0.80 and progress >= 0.85 and coverage >= 0.75 and path_error <= path_threshold and danger_ratio <= 0.20:
        label = 'good'
    elif (completion_rate >= 0.30 or progress >= 0.50) and coverage >= 0.45 and path_error <= 2.0 * path_threshold and danger_ratio <= 0.50:
        label = 'usable'
    else:
        label = 'poor'
    note = (
        f'completion={completion_rate:.2f}, progress={progress:.2f}, coverage={coverage:.2f}, path_error={path_error:.3f}, '
        f'danger_ratio={danger_ratio:.2f}, failure_rate={failure_rate:.2f}'
    )
    return label, note


def info_series(info_episode, key, default=np.nan):
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


def path_coverage_ratios(state_episodes, x_goal, task_config):
    '''Fraction of reference waypoints that the policy trajectory actually visits.'''

    path_threshold = float(task_config.get('path_error_threshold', 0.35))
    coverage_threshold = float(task_config.get('path_coverage_threshold', min(0.2, path_threshold)))
    ref_pos = np.asarray(x_goal)[:, [0, 2, 4]]
    ratios = []
    for states in state_episodes:
        pos = np.asarray(states)[:, [0, 2, 4]]
        if pos.size == 0:
            ratios.append(0.0)
            continue
        distances = np.linalg.norm(ref_pos[:, None, :] - pos[None, :, :], axis=2)
        ratios.append(float(np.mean(np.min(distances, axis=1) <= coverage_threshold)))
    return np.asarray(ratios, dtype=float)


def plot_evaluation(results, x_goal, row, task_config, checkpoint_path, magnitude, args):
    '''Plots one evaluated episode against the path reference and robust metrics.'''
    import matplotlib.pyplot as plt

    n_episodes = len(results['state'])
    episode_idx = min(max(0, args.plot_episode), n_episodes - 1)
    states = np.asarray(results['state'][episode_idx])
    rewards = np.asarray(results['reward'][episode_idx], dtype=float)
    info = results['info'][episode_idx]
    ctrl_freq = float(task_config.get('ctrl_freq', 50))
    path_threshold = float(task_config.get('path_error_threshold', 0.35))

    pos = states[:, [0, 2, 4]]
    ref_pos = np.asarray(x_goal)[:, [0, 2, 4]]
    t_state = np.arange(pos.shape[0]) / ctrl_freq
    t_reward = np.arange(rewards.shape[0]) / ctrl_freq
    t_info = np.arange(len(info)) / ctrl_freq
    path_error = info_series(info, 'path_error')
    progress = info_series(info, 'path_progress_from_start')
    disturbance_active = info_series(info, 'disturbance_active', default=0.0)

    fig = plt.figure(figsize=(13, 9))
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax3d.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2], 'g--', label='reference path')
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'r-', label='policy trajectory')
    ax3d.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color='tab:blue', s=35, label='start')
    ax3d.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color='tab:red', s=35, label='end')
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    ax3d.legend(loc='best')

    ax_xz = fig.add_subplot(2, 2, 2)
    ax_xz.plot(ref_pos[:, 0], ref_pos[:, 2], 'g--', label='reference path')
    ax_xz.plot(pos[:, 0], pos[:, 2], 'r-', label='policy trajectory')
    ax_xz.scatter(pos[0, 0], pos[0, 2], color='tab:blue', s=35, label='start')
    ax_xz.scatter(pos[-1, 0], pos[-1, 2], color='tab:red', s=35, label='end')
    ax_xz.set_xlabel('x')
    ax_xz.set_ylabel('z')
    ax_xz.set_title('x-z projection')
    ax_xz.axis('equal')
    ax_xz.legend(loc='best')

    ax_err = fig.add_subplot(2, 2, 3)
    if path_error.size:
        ax_err.plot(t_info, path_error, color='tab:purple', label='path error')
        ax_err.axhline(path_threshold, color='tab:red', linestyle='--', label='danger threshold')
        shade_disturbance(ax_err, t_info, disturbance_active)
    ax_err.set_xlabel('time (s)')
    ax_err.set_ylabel('path error (m)')
    ax_err.set_title('path error and disturbance window')
    ax_err.legend(loc='best')

    ax_prog = fig.add_subplot(2, 2, 4)
    if progress.size:
        ax_prog.plot(t_info, progress, color='tab:green', label='path progress')
    if rewards.size:
        ax_reward = ax_prog.twinx()
        ax_reward.plot(t_reward, rewards, color='tab:orange', alpha=0.65, label='reward')
        ax_reward.set_ylabel('reward')
        lines, labels = ax_prog.get_legend_handles_labels()
        reward_lines, reward_labels = ax_reward.get_legend_handles_labels()
        ax_prog.legend(lines + reward_lines, labels + reward_labels, loc='best')
    else:
        ax_prog.legend(loc='best')
    shade_disturbance(ax_prog, t_info, disturbance_active)
    ax_prog.set_xlabel('time (s)')
    ax_prog.set_ylabel('normalized progress')
    ax_prog.set_title('progress and reward')

    fig.suptitle(
        f'{os.path.basename(checkpoint_path)} | magnitude={magnitude:.2f} | '
        f'quality={row["quality"]} | {row["quality_note"]}'
    )
    fig.tight_layout()

    saved_path = None
    if args.save_plot:
        saved_path = plot_output_path(args, checkpoint_path, magnitude)
        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
        fig.savefig(saved_path, dpi=180)
        print(f'Saved trajectory plot to {saved_path}')
    if args.plot:
        plt.show()
    else:
        plt.close(fig)
    return saved_path


def evaluate_magnitude(config, checkpoint_path, magnitude, args):
    task_config = set_eval_disturbance(config['task_config'], magnitude, args)
    algo_config = copy.deepcopy(config['algo_config'])
    algo_config.pop('training', None)

    env_func = partial(make, config['task'], **task_config)
    env = env_func(
        seed=args.seed,
        gui=args.gui,
        record=args.record,
        output_dir=args.run_dir,
    )
    ctrl = make(
        config['algo'],
        env_func,
        training=False,
        output_dir=os.path.join(args.run_dir, 'eval_temp'),
        seed=args.seed,
        **algo_config,
    )
    ctrl.load(checkpoint_path)

    experiment = BaseExperiment(env, ctrl)
    seed_episodes = getattr(args, 'seed_episodes', True)
    seeds = [args.seed + i for i in range(args.n_episodes)] if seed_episodes else None
    results, metrics = experiment.run_evaluation(
        n_episodes=args.n_episodes,
        verbose=False,
        seeds=seeds,
        visualization_time_multiplier=args.visualization_speed,
    )
    x_goal = np.array(env.X_GOAL, copy=True)
    coverage = path_coverage_ratios(results['state'], x_goal, task_config)
    experiment.close()

    row = {
        'checkpoint': os.path.basename(checkpoint_path),
        'magnitude': magnitude,
        'average_return': scalar(metrics, 'average_return'),
        'average_rmse': scalar(metrics, 'average_rmse'),
        'average_path_error': scalar(metrics, 'average_path_error'),
        'average_peak_path_error': scalar(metrics, 'average_peak_path_error'),
        'average_final_progress': scalar(metrics, 'average_final_progress'),
        'average_path_coverage': float(np.nanmean(coverage)) if coverage.size else np.nan,
        'completion_rate': scalar(metrics, 'completion_rate'),
        'average_danger_zone_steps': scalar(metrics, 'average_danger_zone_steps'),
        'average_recovery_time': scalar(metrics, 'average_recovery_time'),
        'average_peak_deviation_after_disturbance': scalar(metrics, 'average_peak_deviation_after_disturbance'),
        'failure_rate': scalar(metrics, 'failure_rate'),
    }
    row['quality'], row['quality_note'] = quality_assessment(row, task_config)
    if args.plot or args.save_plot:
        plot_path = plot_evaluation(results, x_goal, row, task_config, checkpoint_path, magnitude, args)
        if plot_path is not None:
            row['plot_path'] = plot_path
    return row


def main():
    args = parse_args()
    config = load_run_config(args.run_dir)
    ckpt_path = checkpoint_path(args)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Model checkpoint not found: {ckpt_path}')

    out_path = output_path(args, ckpt_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f'Checkpoint: {ckpt_path}')
    print(f'Output CSV: {out_path}')

    rows = []
    for magnitude in args.magnitudes:
        row = evaluate_magnitude(config, ckpt_path, magnitude, args)
        rows.append(row)
        print(
            'mag={magnitude:.2f} return={average_return:.3f} path_error={average_path_error:.3f} '
            'peak={average_peak_path_error:.3f} progress={average_final_progress:.3f} coverage={average_path_coverage:.3f} '
            'completion={completion_rate:.2f} danger_steps={average_danger_zone_steps:.1f} '
            'quality={quality} ({quality_note})'.format(**row)
        )

    with open(out_path, 'w', newline='', encoding='UTF-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved summary to {out_path}')


if __name__ == '__main__':
    main()
