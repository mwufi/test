from unittest import TestCase
from configs.parser import load_config, load_yaml
from box.exceptions import BoxKeyError

class Test(TestCase):

    yaml_file = 'sample.yml'

    def test_can_read(self):
        config = load_config(self.yaml_file)

        self.assertEqual(config.data.loader.workers, 20)
        self.assertEqual(config.name, 'random-run')

    def test_not_exists(self):
        config = load_config(self.yaml_file)

        self.assertFalse('random_stuff' in config)
        with self.assertRaises(BoxKeyError):
            print(config.random_stuff)

    def test_can_write(self):
        config = load_config(self.yaml_file)

        config.foo = 2

        self.assertEqual(config.foo, 2)

    def test_can_nested_write(self):
        config = load_config(self.yaml_file)

        config.data.name = 'wando'
        self.assertEqual(config.data.name, 'wando')
        self.assertEqual(config.data['clone_again'], False)

    def test_to_json(self):
        config = load_config(self.yaml_file)

        output = config.to_json('hola.json')
        output = config.to_yaml('copy.yml')

        # Test that we can read from the copy again
        config = load_config('copy.yml')

        self.assertEqual(config.data.loader.workers, 20)
        self.assertEqual(config.name, 'random-run')

