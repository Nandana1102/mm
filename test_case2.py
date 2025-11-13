import unittest

class TestPalindrome(unittest.TestCase):
    def test_palindrome(self):
        self.assertTrue("madam" == "madam"[::-1])
        self.assertFalse("hello" == "hello"[::-1])

if __name__ == "__main__":
    unittest.main()
