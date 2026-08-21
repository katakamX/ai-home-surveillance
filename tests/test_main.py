import unittest

from src.main import main


class TestMain(unittest.TestCase):
    def test_main_is_callable(self):
        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()
