import pandas as pd
from pathlib import Path

from amber.database import MultiDataConfig, BackendConfig, extract_data

def build_data_cfg(outputdir):
    dts = [
        {
            "name":"MSEEL_3H",
            "category":"event",
            "role":"multi",
        },
        {
            "name":"Clearfield_mw4",
            "category":"event",
            "role":"multi",
        },
        {
            "name":"PNR-1",
            "category":"event",
            "minppicks":16,
            "minspicks":16,
            "plen":0.02,
            "slen":0.04,
            "role":"test",
        },
        {
            "name":"MSEEL_3H",
            "category":"noise",
            "role":"multi",
        },
        {
            "name":"Clearfield_mw4",
            "category":"noise",
            "role":"multi",
        },
        {
            "name":"PNR-1",
            "category":"noise",
            "role":"multi",
        },
    ]

    return MultiDataConfig(
        datasets=dts,
        outputdir=outputdir,
        batchsize=5
    )

    
def test_extract(tmp_path):
    be_cfg = BackendConfig(
        is_zip=True,
        zipfilepath=Path(__file__).parent/"amber_raw_segys_mini.zip"
    )

    data_cfg = build_data_cfg(tmp_path)
    extract_data(data_cfg, be_cfg)

    csv_path = tmp_path / "metadata.csv"
    h5_path = tmp_path / "waveforms.hdf5"

    assert csv_path.exists()
    assert h5_path.exists()

    df = pd.read_csv(csv_path)
    
    assert (df["trace_component_order"].astype(str) == "NEZ").all()
    assert (df["dimension_order"].astype(str) == "CW").all()
    assert not df["trace_start_time"].astype(str).isna().any()
    
    ndp = df["trace_name"].str.split(":").str[-1].astype(int)
    pickcols = df.filter(like="arrival_sample")
    bounded = (pickcols >= 0) & (pickcols < ndp.to_numpy()[:, None])

    assert (pickcols.isna() | bounded).to_numpy().all()