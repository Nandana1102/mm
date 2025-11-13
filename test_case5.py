import unittest
from sympy import fibonacci

class TestFibonacci(unittest.TestCase):
    def test_fib(self):
        self.assertEqual([fibonacci(i) for i in range(5)], [0,1,1,2,3])

if __name__ == "__main__":
    unittest.main()
