import unittest
from sympy import isprime

class TestPrime(unittest.TestCase):
    def test_primes(self):
        self.assertTrue(isprime(7))
        self.assertFalse(isprime(9))
        self.assertFalse(isprime(-3))

if __name__ == "__main__":
    unittest.main()
