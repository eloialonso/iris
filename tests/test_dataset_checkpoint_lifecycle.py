"""Checkpoint disk contents must match live replay episodes after eviction/clear."""
import torch

from dataset import EpisodesDataset, EpisodesDatasetRamMonitoring
from episode import Episode


def episode(value, length=1):
    return Episode(
        observations=torch.full((length, 1, 4, 4), value, dtype=torch.uint8),
        actions=torch.zeros(length, dtype=torch.long),
        rewards=torch.full((length,), float(value)),
        ends=torch.zeros(length, dtype=torch.long),
        mask_padding=torch.ones(length, dtype=torch.bool),
    )


def checkpoint_values(directory):
    restored = EpisodesDataset()
    restored.load_disk_checkpoint(directory)
    return {episode_id: int(restored.get_episode(episode_id).observations[0, 0, 0, 0])
            for episode_id in restored.episode_id_to_queue_idx}


def test_unsaved_eviction_checkpoints_only_remaining_episodes(tmp_path):
    dataset = EpisodesDataset(max_num_episodes=2)
    for value in [10, 11, 12]:
        dataset.add_episode(episode(value))
    dataset.update_disk_checkpoint(tmp_path)
    assert checkpoint_values(tmp_path) == {1: 11, 2: 12}
    assert not dataset.newly_modified_episodes
    assert not dataset.newly_deleted_episodes


def test_modified_persisted_episode_can_be_evicted_before_next_save(tmp_path):
    dataset = EpisodesDataset(max_num_episodes=1)
    dataset.add_episode(episode(7))
    dataset.update_disk_checkpoint(tmp_path)
    dataset.update_episode(0, episode(8))
    dataset.add_episode(episode(9))
    dataset.update_disk_checkpoint(tmp_path)
    assert checkpoint_values(tmp_path) == {1: 9}


def test_repeated_unsaved_turnover_does_not_leave_stale_disk_episodes(tmp_path):
    dataset = EpisodesDataset(max_num_episodes=3)
    for value in range(50):
        dataset.add_episode(episode(value))
    dataset.update_disk_checkpoint(tmp_path)
    assert checkpoint_values(tmp_path) == {47: 47, 48: 48, 49: 49}


def test_clear_removes_persisted_old_episodes_from_next_checkpoint(tmp_path):
    dataset = EpisodesDataset()
    dataset.add_episode(episode(1))
    dataset.add_episode(episode(2))
    dataset.update_disk_checkpoint(tmp_path)
    dataset.clear()
    dataset.add_episode(episode(3))
    dataset.update_disk_checkpoint(tmp_path)
    assert checkpoint_values(tmp_path) == {2: 3}


def test_clear_before_first_save_discards_stale_dirty_entries(tmp_path):
    dataset = EpisodesDataset()
    dataset.add_episode(episode(1))
    dataset.add_episode(episode(2))
    dataset.clear()
    dataset.add_episode(episode(3))
    dataset.update_disk_checkpoint(tmp_path)
    assert checkpoint_values(tmp_path) == {2: 3}


def test_already_missing_evicted_file_does_not_prevent_checkpoint(tmp_path):
    dataset = EpisodesDataset(max_num_episodes=1)
    dataset.add_episode(episode(1))
    dataset.update_disk_checkpoint(tmp_path)
    (tmp_path / '0.pt').unlink()
    dataset.add_episode(episode(2))
    dataset.update_disk_checkpoint(tmp_path)
    assert checkpoint_values(tmp_path) == {1: 2}


def test_ram_monitoring_eviction_preserves_checkpoint_lifecycle(tmp_path):
    dataset = EpisodesDatasetRamMonitoring(max_ram_usage='100G')
    dataset.max_num_steps = 3
    dataset.add_episode(episode(1, length=2))
    dataset.add_episode(episode(2, length=2))
    assert dataset.num_steps == 2
    dataset.update_disk_checkpoint(tmp_path)
    assert checkpoint_values(tmp_path) == {1: 2}


def test_normal_checkpoint_updates_live_episode_data(tmp_path):
    dataset = EpisodesDataset()
    dataset.add_episode(episode(1))
    dataset.update_disk_checkpoint(tmp_path)
    dataset.update_episode(0, episode(2))
    dataset.update_disk_checkpoint(tmp_path)
    state = torch.load(tmp_path / '0.pt')
    torch.testing.assert_close(state['rewards'], torch.tensor([1., 2.]))
    assert checkpoint_values(tmp_path) == {0: 1}


def test_empty_checkpoint_loads_as_empty_dataset(tmp_path):
    dataset = EpisodesDataset()
    dataset.update_disk_checkpoint(tmp_path)
    restored = EpisodesDataset()
    restored.load_disk_checkpoint(tmp_path)
    assert len(restored) == 0
    assert restored.num_seen_episodes == 0


def test_clear_checkpoint_can_roundtrip_without_adding_an_episode(tmp_path):
    dataset = EpisodesDataset()
    dataset.add_episode(episode(1))
    dataset.update_disk_checkpoint(tmp_path)
    dataset.clear()
    dataset.update_disk_checkpoint(tmp_path)
    assert list(tmp_path.iterdir()) == []
    restored = EpisodesDataset()
    restored.load_disk_checkpoint(tmp_path)
    restored.add_episode(episode(2))
    restored.update_disk_checkpoint(tmp_path)
    assert checkpoint_values(tmp_path) == {0: 2}


def test_clear_then_reload_discards_pending_deletions(tmp_path):
    dataset = EpisodesDataset()
    dataset.add_episode(episode(1))
    dataset.add_episode(episode(2))
    dataset.update_disk_checkpoint(tmp_path)
    dataset.clear()
    dataset.load_disk_checkpoint(tmp_path)
    dataset.update_disk_checkpoint(tmp_path)
    assert checkpoint_values(tmp_path) == {0: 1, 1: 2}


def test_restored_ram_step_count_preserves_eviction_budget(tmp_path):
    original = EpisodesDatasetRamMonitoring(max_ram_usage='100G')
    original.max_num_steps = 3
    original.add_episode(episode(1, length=2))
    original.update_disk_checkpoint(tmp_path)
    restored = EpisodesDatasetRamMonitoring(max_ram_usage='100G')
    restored.max_num_steps = 3
    restored.load_disk_checkpoint(tmp_path)
    assert restored.num_steps == 2
    restored.add_episode(episode(2, length=2))
    assert restored.num_steps == 2
    restored.update_disk_checkpoint(tmp_path)
    assert checkpoint_values(tmp_path) == {1: 2}
