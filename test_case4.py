import unittest

class TestSort(unittest.TestCase):
    def test_sorted(self):
        self.assertEqual(sorted([3,1,2]), [1,2,3])
        self.assertEqual(sorted([]), [])
        self.assertEqual(sorted([4,2,4,1]), [1,2,4,4])

if __name__ == "__main__":
    unittest.main()
