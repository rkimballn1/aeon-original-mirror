import tempfile
import numpy as np
from PIL import Image as PILImage
import pytest
import os
import math
import glob
import json

from aeon import DataLoader, dict2json
from mock_data import random_manifest, generic_config, invalid_image

batch_size = 2

def test_loader_invalid_config_type():
    manifest = random_manifest(10)
    config = generic_config(manifest.name, batch_size)

    config["etl"][0]["type"] = 'invalid type name'

    with pytest.raises(RuntimeError) as ex:
        dl = DataLoader(config)
    assert 'unsupported' in str(ex)


def test_loader_missing_config_field():
    manifest = random_manifest(10)
    config = generic_config(manifest.name, batch_size)

    del config['etl'][0]["height"]

    with pytest.raises(RuntimeError) as ex:
        dl = DataLoader(config)
    assert 'height' in str(ex)


def test_loader_non_existant_manifest():
    config = generic_config('/this_manifest_file_does_not_exist', batch_size)

    with pytest.raises(RuntimeError) as ex:
        dl = DataLoader(config)
    assert "doesn't exist" in str(ex)


def test_loader_invalid_manifest():
    filename = tempfile.mkstemp()[1]
    config = generic_config(invalid_image(filename), batch_size)

    with pytest.raises(Exception) as ex:
        dl = DataLoader(config)
    assert 'must be string, but is null' in str(ex)


def test_loader():
    # NOTE: manifest needs to stay in scope until DataLoader has read it.
    for i in range(1, 10):
        manifest = random_manifest(i)
        config = generic_config(manifest.name, batch_size)

        dl = DataLoader(config)

        assert len(list(iter(dl))) == math.ceil(float(i)/batch_size)


def test_loader_repeat_iter():
    # NOTE: manifest needs to stay in scope until DataLoader has read it.
    manifest = random_manifest(10)
    config = generic_config(manifest.name, batch_size)

    dl = DataLoader(config)

    assert len(list(iter(dl))) == math.ceil(10./batch_size)


def test_loader_exception_next():
    # NOTE: manifest needs to stay in scope until DataLoader has read it.
    cwd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path+'/test_data')
    manifest = open("manifest.csv")

    config = generic_config(manifest.name, batch_size)
    dl = DataLoader(config)
    num_of_batches_in_manifest = 60
    for x in range(0, num_of_batches_in_manifest):
        next(dl)
    with pytest.raises(StopIteration) as ex:
        next(dl)
    manifest.close()
    os.chdir(cwd)


def test_loader_exception_iter():
    # NOTE: manifest needs to stay in scope until DataLoader has read it.
    cwd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path+'/test_data')
    manifest = open("manifest.csv")

    config = generic_config(manifest.name, batch_size)
    dl = DataLoader(config)

    num_of_manifest_entries = 120.
    assert len(list(iter(dl))) == math.ceil(num_of_manifest_entries/batch_size)

    manifest.close()
    os.chdir(cwd)


def test_loader_reset():
    # NOTE: manifest needs to stay in scope until DataLoader has read it.
    manifest = random_manifest(10)
    config = generic_config(manifest.name, batch_size)
    dl = DataLoader(config)
    assert len(list(iter(dl))) == math.ceil(10./batch_size)
    dl.reset()
    assert len(list(iter(dl))) == math.ceil(10./batch_size)


def test_loader_json_parser_fail():
    files = glob.glob("./json/fail*.json")

    for f in files:
        with open(f) as json_file:
            json_string = json_file.read()

        try:
            config = json.loads(json_string)
        except ValueError:
            continue

        json_string = '{"config": %s}' % json_string
        config = json.loads(json_string)
        with pytest.raises(RuntimeError) as ex:
            dl = DataLoader(config)
        assert 'Required Argument' in str(ex) 


def test_loader_json_parser_pass():
    files = glob.glob("./json/pass*.json")

    for f in files:
        with open(f) as json_file:
            json_string = json_file.read()
            # config must be a dict so make sure it is a dict
            json_string = '{"config": %s}' % json_string
        config = json.loads(json_string)
        with pytest.raises(RuntimeError) as ex:
            dl = DataLoader(config)
        assert 'Required Argument' in str(ex) 


def test_parser_dump_pass():
    files = glob.glob("./json/pass*.json")

    for f in files:
        with open(f) as json_file:
            json_string = json_file.read()
        config = json.loads(json_string)
        # it should not throw exception unless config is not a dictionary
        if isinstance(config, dict):
            config2 = json.loads(dict2json(config))
            assert (config == config2)
        else:
            with pytest.raises(RuntimeError) as ex:
                dict2json(config)
            assert("can only take dictionary" in str(ex))


def test_parser_dump_fail():
    files = glob.glob("./json/fail*.json")

    for f in files:
        with open(f) as json_file:
            json_string = json_file.read()
        try:
            config = json.loads(json_string)
        except ValueError:
            continue

        with pytest.raises(RuntimeError) as ex:
            dict2json(config)
        assert("can only take dictionary" in str(ex))


def test_parse_json_dict_list_pass():
    test_dir = os.path.dirname(os.path.realpath(__file__)) + '/test_data/'

    config = {'batch_size': 16, 'manifest_root': test_dir, 'manifest_filename': test_dir + 'manifest.csv',
        'etl': [{'type': 'image', 'width': 32, 'height': 32}, {'type': 'label', 'binary': False}]}

    dl = DataLoader(config)
    assert (dl.config["etl"][0]["type"] == 'image' and dl.config["etl"][1]["type"] == 'label')


def test_parse_json_dict_tuple_pass():
    test_dir = os.path.dirname(os.path.realpath(__file__)) + '/test_data/'

    config = {'batch_size': 16, 'manifest_root': test_dir, 'manifest_filename': test_dir + 'manifest.csv',
        'etl': ({'type': 'image', 'width': 32, 'height': 32}, {'type': 'label', 'binary': False})}

    dl = DataLoader(config)
    assert (dl.config["etl"][0]["type"] == 'image' and dl.config["etl"][1]["type"] == 'label')


if __name__ == '__main__':
    pytest.main()
