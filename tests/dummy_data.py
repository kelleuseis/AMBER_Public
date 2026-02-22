import numpy as np
import pandas as pd
import h5py

from amber.dataloaders import AMBER, DatasetConfig

dummy_ndps = [2000, 2500, 3000, 4000]
dummy_nsta = [6, 8, 8, 2]
dummy_evlist = [
    # p base sample, s base sample, station sample delay, event type
    [
        (500, 1500, 0, 'earthquake')
    ],
    [
        (15, 50, 0, 'earthquake'),
        (2350, 2450, 0, 'earthquake'),
        (np.nan, np.nan, 0, 'noise'),
        (np.nan, np.nan, 0, 'earthquake')
    ],
    [
        (15, 2950, 0, 'earthquake'),
        (1500, 2000, 5, 'earthquake')
    ],
    [
        (2000, 3000, 0, 'earthquake')
    ],
]
dummy_sf = [1000, 8000, 8000, 4000]

expected_sample_ids = {
    "dummy_0_0",
    "dummy_1_0",
    "dummy_1_1",
    "dummy_1_2",
    "dummy_1_3",
    "dummy_2_0",
    "dummy_2_1",
}



dummy_mdl_ndps = 2500
dummy_mdl_nsta = 4



def create_dummy_hdf5(dirpath):
    h5_path = dirpath / "waveforms.hdf5"

    with h5py.File(h5_path, "w") as f:
        traces_grp = f.require_group("data")
        meta_grp = f.require_group("data_format")

        for bktidx in range(len(dummy_ndps)):
            dset = traces_grp.create_dataset(
                f"bucket{bktidx}",
                shape=(
                    dummy_nsta[bktidx]*len(dummy_evlist[bktidx]), 3, dummy_ndps[bktidx]
                ),
                maxshape=(None, 3, dummy_ndps[bktidx]),
                dtype="float32",
                chunks=(1, 3, dummy_ndps[bktidx]),
                compression="gzip"
            )
            dset[:] = np.random.randn(
                dummy_nsta[bktidx]*len(dummy_evlist[bktidx]), 3, dummy_ndps[bktidx]
            ).astype(np.float32)


        meta_grp.create_dataset(
            "component_order",
            data="NEZ",
            dtype=h5py.string_dtype("utf-8")
        )
        meta_grp.create_dataset(
            "dimension_order",
            data="CW",
            dtype=h5py.string_dtype("utf-8")
        )
        meta_grp.create_dataset(
            "bucket_ndp",
            data=np.array(dummy_ndps, dtype=np.int32)
        )

    return h5_path


def create_dummy_csv(dirpath):
    csv_path = dirpath / "metadata.csv"
    rows = []

    for bktidx in range(len(dummy_ndps)):
        rowidx = 0

        for evidx, (pbase, sbase, spread, typ) in enumerate(dummy_evlist[bktidx]):
            for staidx in range(dummy_nsta[bktidx]):
                p_arr = pbase + staidx*spread
                s_arr = sbase + staidx*spread

                rows.append({
                    "trace_name": f"bucket{bktidx}${rowidx},:3,:{dummy_ndps[bktidx]}",
                    "split": "train",
                    "station": f"{staidx:03d}",
                    "dataset": "dummy",
                    "event_id": f"dummy_{bktidx}_{evidx}",
                    "trace_sampling_rate_hz": dummy_sf[bktidx],
                    "trace_category": typ,
                    "trace_start_time": "1970-01-01T00:00:00+00:00",
                    "trace_p_arrival_sample": p_arr,
                    "trace_s_arrival_sample": s_arr,
                    "trace_p_length": 0.02,
                    "trace_s_length": 0.02,
                    "trace_component_order": "NEZ",
                    "dimension_order": "CW",
                })

                rowidx += 1

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    return csv_path


class DummyLabeller:
    def __call__(self, pickarr, eventdf, stnout, window_idx, samplerate):
        return [pickarr]

def create_dummy_dataset(h5_path, csv_path):
    config = DatasetConfig(
        windowlength=dummy_mdl_ndps,
        nstation=dummy_mdl_nsta,
        fullphasecoverage=True,
        normalisation="tracewise",
        sequentialstations=False,
    )
    dts = AMBER(
        config=config,
        h5_path=h5_path,
        csv_path=csv_path,
        labeller=DummyLabeller(),
        augmentations=None,
        mode="train",
    )
    return dts