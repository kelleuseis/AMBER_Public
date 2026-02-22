import pytest
import torch
import numpy as np

from dummy_data import expected_sample_ids
from amber.Augmentations.base_augmentation import BaseAugmentation


def test_dataset_init(dummy_dataset):
    assert set(dummy_dataset.sample_ids) == expected_sample_ids


def test_extract_sample(dummy_dataset):
    trace_grp = dummy_dataset._get_h5()

    for idx in range(len(dummy_dataset.sample_ids)):
        sid = dummy_dataset.sample_ids[idx]

        waves_raw, eventdf = dummy_dataset.extract_sample(sid, trace_grp)

        assert waves_raw.ndim == 3
        assert waves_raw.shape[1] == 3
        assert waves_raw.dtype == np.float32
        assert waves_raw.shape[0] == len(eventdf)


def test_extract_window(dummy_dataset):
    trace_grp = dummy_dataset._get_h5()

    for idx in range(len(dummy_dataset.sample_ids)):
        sid = dummy_dataset.sample_ids[idx]

        waves_raw, eventdf = dummy_dataset.extract_sample(sid, trace_grp)
        waves, pickarr, stnout, window_idx = dummy_dataset.extract_window(
            waves_raw, eventdf
        )
        
        confignsta = dummy_dataset.config.nstation
        configlen = dummy_dataset.config.windowlength
        rawlen = waves_raw.shape[-1]

        assert waves.ndim == 3
        assert waves.shape[1] == 3
        assert waves.shape[-1] == configlen

        assert waves.shape[0] == confignsta
        assert pickarr.shape[0] == confignsta
        assert len(stnout) == confignsta
        
        reallen = rawlen if rawlen >= configlen else rawlen + configlen
        assert 0 <= window_idx <= (reallen - configlen)

        assert waves.dtype == np.float32
        assert pickarr.dtype == np.float32
        
        if sid == "dummy_1_0":
            exp_pickarr = np.zeros((confignsta, 2, 1), dtype=np.float32)
            exp_pickarr[:, 0, 0] = 15
            exp_pickarr[:, 1, 0] = 50
            np.testing.assert_allclose(pickarr, exp_pickarr)

        if sid == "dummy_2_0":
            exp_pickarr = np.zeros((confignsta, 2, 1), dtype=np.float32)
            exp_pickarr[:, 0, 0] = np.nan
            exp_pickarr[:, 1, 0] = 2450
            np.testing.assert_allclose(pickarr, exp_pickarr)
            


def test_corrupted_trace_name(dummy_dataset, monkeypatch):
    trace_grp = dummy_dataset._get_h5()
    bad_sid = dummy_dataset.sample_ids[0]

    corrupted_meta = dummy_dataset.meta.copy()
    rowidxs = corrupted_meta.index[
        corrupted_meta["event_id"] == bad_sid
    ]
    corrupted_meta.loc[rowidxs[0], "trace_name"] = "invalid"

    monkeypatch.setattr(dummy_dataset, "meta", corrupted_meta)

    with pytest.raises(ValueError):
        dummy_dataset.extract_sample(bad_sid, trace_grp)
        
        
        
class DummyAug(BaseAugmentation):
    scope = "raw"
    def augment_raw(self, waves_all, eventdf, samplerate):
        waves_all = waves_all.copy()
        waves_all[0] = 0.0
        return waves_all, samplerate

def test_zeroed_trace(dummy_dataset):
    dummy_dataset.augmentations = [DummyAug({})]
    trace_grp = dummy_dataset._get_h5()

    waves, *_ = dummy_dataset[0]

    assert not torch.isnan(waves).any()
    assert not torch.isinf(waves).any()
    assert waves.dtype == torch.float32
    