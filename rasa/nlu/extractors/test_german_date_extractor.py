import unittest
from duckling_entity_extractor import GermanDatetimeEntityExtractor

class TestGermanDatetimeExtractor(unittest.TestCase):
    def setUp(self):
        config = {"locale": "de"}
        self.extractor = GermanDatetimeEntityExtractor(config)

        self.config_en = {"locale": "en"}
        self.extractor_en = GermanDatetimeEntityExtractor(self.config_en)
    
    def test_modify_datetime_format(self):
        original_result = [
            {"dim": "time", "value": "2023-10-20"},
        ]
        
        modified_result = self.extractor.modify_datetime_format(original_result)
        
        expected_result = [
            {"dim": "time", "value": "20-10-2023"},
        ]
        
        self.assertEqual(modified_result, expected_result)
    
    def test_modify_datetime_format_en(self):
        original_result = [
            {"dim": "time", "value": "2023-10-20T14:30:00.000Z"}
        ]
        
        modified_result = self.extractor_en.modify_datetime_format(original_result)
        
        expected_result = [
            {"dim": "time", "value": "2023-10-20T14:30:00.000Z"}
        ]
        
        self.assertEqual(modified_result, expected_result)

if __name__ == "__main__":
    unittest.main()