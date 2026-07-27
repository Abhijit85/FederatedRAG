import unittest
from pathlib import Path

from scripts.audit_table14_discrepancy import central_benchmark_rows, extract_textgrad_defaults
from scripts.run_structured_payload_control import parse_seed_list, slugify_condition


class RebuttalSupportTest(unittest.TestCase):
    def test_parse_seed_list(self):
        self.assertEqual(parse_seed_list('1, 2,5'), [1, 2, 5])

    def test_slugify_condition(self):
        self.assertEqual(slugify_condition('typed'), 'typed')
        self.assertEqual(slugify_condition('field-untyped'), 'field_untyped')

    def test_extract_textgrad_defaults(self):
        defaults = extract_textgrad_defaults(Path('scripts/run_fed_textgrad.py'))
        self.assertEqual(defaults['aggregate_method'], 'summarization')
        self.assertEqual(defaults['batch_size'], 3)
        self.assertEqual(defaults['max_steps'], 3)
        self.assertEqual(defaults['rounds'], 1)

    def test_central_benchmark_rows_filters_to_target_benchmark(self):
        records = [
            {
                'timestamp': '2025-11-14T05:43:05.836879Z',
                'section': 'central',
                'payload': {
                    'mixed_queries': 'bbh_object_counting_eval_v3.json',
                    'overall': {'accuracy': 0.92, 'correct': 46, 'total': 50},
                },
            },
            {
                'timestamp': '2026-07-24T07:12:45.145417Z',
                'section': 'central',
                'payload': {
                    'mixed_queries': 'math_only.json',
                    'overall': {'accuracy': 0.1, 'correct': 1, 'total': 10},
                },
            },
        ]
        rows = central_benchmark_rows(records, 'bbh_object_counting_eval_v3.json')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['accuracy'], 0.92)
        self.assertEqual(rows[0]['correct'], 46)
        self.assertEqual(rows[0]['total'], 50)


if __name__ == '__main__':
    unittest.main()
