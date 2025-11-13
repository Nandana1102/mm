import unittest
import math
from sympy import *

class TestFactorial(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(math.factorial(5),120)
    
    def test_zero(self):
        self.assertEqual(math.factorial(0),1)
    
    # def test_negative(self):
    #     self.assertIsNone(-3)

if __name__ == "__main__":
    unittest.main()