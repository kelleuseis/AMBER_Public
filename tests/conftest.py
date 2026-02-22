import pytest

from dummy_data import (
    create_dummy_hdf5,
    create_dummy_csv,
    create_dummy_dataset
)


@pytest.fixture(scope="session")
def dummy_hdf5(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("data")
    h5_path = create_dummy_hdf5(tmp_dir)
    return h5_path

@pytest.fixture(scope="session")
def dummy_csv(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("data")
    csv_path = create_dummy_csv(tmp_dir)
    return csv_path

@pytest.fixture
def dummy_dataset(dummy_hdf5, dummy_csv):
    dts = create_dummy_dataset(dummy_hdf5, dummy_csv)
    return dts
