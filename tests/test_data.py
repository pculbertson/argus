from argus.data import CameraCubePoseDataset, CameraCubePoseDatasetConfig

# ######### #
# UTILITIES #
# ######### #


def run_assertions(dataset, expected_len) -> None:
    """A helper function to run assertions on the items in the dataset."""
    assert (
        len(dataset) == expected_len
    ), f"The length of the dataset is incorrect! Expected {expected_len}, got {len(dataset)}"
    example = dataset[0]
    assert set(example.keys()) == {"images", "cube_pose"}, "The keys of the example are incorrect!"
    assert example["images"].shape == (2 * 3, 376, 672), "The shape of the images is incorrect!"
    assert example["cube_pose"].shape == (7,), "The shape of the cube poses is incorrect!"


# ##### #
# TESTS #
# ##### #


def test_len(dummy_data_path) -> None:
    """Tests the __len__ method of the dataset."""
    # load the dataset
    cfg = CameraCubePoseDatasetConfig(dummy_data_path)
    dataset = CameraCubePoseDataset(cfg, train=True)
    run_assertions(dataset, 10)

    cfg = CameraCubePoseDatasetConfig(dummy_data_path)
    dataset = CameraCubePoseDataset(cfg, train=False)
    run_assertions(dataset, 5)


def test_get_item(dummy_data_path) -> None:
    """Tests the __getitem__ method of the dataset."""
    # load the dataset
    cfg = CameraCubePoseDatasetConfig(dummy_data_path)
    dataset = CameraCubePoseDataset(cfg, train=True)
    run_assertions(dataset, 10)

    cfg = CameraCubePoseDatasetConfig(dummy_data_path)
    dataset = CameraCubePoseDataset(cfg, train=False)
    run_assertions(dataset, 5)
