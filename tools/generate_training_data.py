#!/usr/bin/env python3
"""
Script to generate training data automatically
Usage: python generate_training_data.py
"""

import random
from typing import List, Dict
import yaml


# ============ DATA GENERATORS ============

class TrainingDataGenerator:
    """Generate diverse training examples"""

    def __init__(self):
        self.templates = {
            'search_product': [
                "tìm {product}",
                "cho tôi xem {product}",
                "có {product} nào không",
                "tôi muốn mua {product}",
                "{product} giá bao nhiêu",
                "tìm {product} {brand}",
                "{product} {brand} có không",
                "cho xem {product} dưới {price}",
                "tìm {product} cho {use_case}",
                "{product} {use_case}",
            ],
            'ask_price': [
                "giá bao nhiêu",
                "{product} giá bao nhiêu",
                "giá {product} {brand}",
                "bao nhiêu tiền",
                "hỏi giá {product}",
                "giá cả thế nào",
                "{product} này giá bn",
            ],
            'compare_products': [
                "so sánh {product1} và {product2}",
                "{product1} so với {product2}",
                "khác nhau gì giữa {brand1} và {brand2}",
                "{brand1} tốt hơn {brand2} không",
                "nên chọn {product} nào",
            ],
        }

        self.products = ['laptop', 'máy tính', 'máy chủ', 'server', 'camera', 'switch']
        self.brands = ['Dell', 'HP', 'ASUS', 'MSI', 'Lenovo', 'Cisco', 'Hikvision']
        self.prices = ['10 triệu', '20 triệu', '30 triệu', '50 triệu', '100 triệu']
        self.use_cases = ['gaming', 'văn phòng', 'sinh viên', 'doanh nghiệp', 'thiết kế']

    def generate_intent_examples(self, intent_name: str, num_examples: int = 50) -> List[str]:
        """Generate examples for an intent"""
        if intent_name not in self.templates:
            print(f"Warning: No template for intent '{intent_name}'")
            return []

        examples = []
        templates = self.templates[intent_name]

        for _ in range(num_examples):
            template = random.choice(templates)

            # Replace placeholders
            example = template
            if '{product}' in template:
                example = example.replace('{product}', random.choice(self.products))
            if '{brand}' in template:
                example = example.replace('{brand}', random.choice(self.brands))
            if '{price}' in template:
                example = example.replace('{price}', random.choice(self.prices))
            if '{use_case}' in template:
                example = example.replace('{use_case}', random.choice(self.use_cases))
            if '{product1}' in template:
                example = example.replace('{product1}', random.choice(self.products))
            if '{product2}' in template:
                example = example.replace('{product2}', random.choice(self.products))
            if '{brand1}' in template:
                example = example.replace('{brand1}', random.choice(self.brands))
            if '{brand2}' in template:
                example = example.replace('{brand2}', random.choice(self.brands))

            # Avoid duplicates
            if example not in examples:
                examples.append(example)

        return examples

    def generate_nlu_file(self, output_path: str):
        """Generate complete NLU file"""
        nlu_data = {
            'version': '3.1',
            'nlu': []
        }

        for intent_name in self.templates.keys():
            examples = self.generate_intent_examples(intent_name, num_examples=50)
            examples_text = '\n      - '.join([''] + examples)

            nlu_data['nlu'].append({
                'intent': intent_name,
                'examples': examples_text
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(nlu_data, f, allow_unicode=True, sort_keys=False)

        print(f"✅ Generated NLU file: {output_path}")
        print(f"   Total intents: {len(self.templates)}")
        print(f"   Total examples: {sum(len(self.generate_intent_examples(i, 50)) for i in self.templates.keys())}")


# ============ ENTITY ANNOTATOR ============

class EntityAnnotator:
    """Automatically annotate entities in examples"""

    def __init__(self):
        self.entity_patterns = {
            'product_type': ['laptop', 'máy tính', 'máy chủ', 'server', 'camera', 'switch'],
            'brand': ['Dell', 'HP', 'ASUS', 'MSI', 'Lenovo', 'Cisco', 'Hikvision', 'Dahua'],
            'price_range': ['10 triệu', '20 triệu', '30 triệu', 'dưới 50 triệu'],
        }

    def annotate(self, text: str) -> str:
        """Annotate entities in text"""
        annotated = text

        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                if pattern in annotated and f'[{pattern}]' not in annotated:
                    annotated = annotated.replace(pattern, f'[{pattern}]({entity_type})')

        return annotated

    def annotate_file(self, input_path: str, output_path: str):
        """Annotate entities in an NLU file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if 'nlu' not in data:
            print("Error: Not a valid NLU file")
            return

        for intent_data in data['nlu']:
            if 'examples' in intent_data:
                examples = intent_data['examples'].split('\n')
                annotated_examples = []

                for example in examples:
                    if example.strip().startswith('- '):
                        text = example.strip()[2:]  # Remove '- '
                        annotated = self.annotate(text)
                        annotated_examples.append(f"      - {annotated}")
                    else:
                        annotated_examples.append(example)

                intent_data['examples'] = '\n'.join(annotated_examples)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)

        print(f"✅ Annotated file saved: {output_path}")


# ============ DATA AUGMENTER ============

class DataAugmenter:
    """Augment existing training data"""

    def __init__(self):
        self.variations = {
            'laptop': ['máy tính', 'máy tính xách tay', 'notebook', 'laptop'],
            'máy chủ': ['server', 'máy chủ', 'may chu'],
            'camera': ['cam', 'camera', 'camera an ninh'],
        }

        self.typos = {
            'bao nhiêu': ['bao nhiu', 'bn'],
            'tìm': ['tim', 'tiem'],
            'có': ['co'],
        }

    def augment_examples(self, examples: List[str]) -> List[str]:
        """Create variations of examples"""
        augmented = examples.copy()

        for example in examples[:]:  # Copy to avoid modifying during iteration
            # Synonym replacement
            for original, variations in self.variations.items():
                if original in example:
                    for variation in variations:
                        new_example = example.replace(original, variation)
                        if new_example not in augmented:
                            augmented.append(new_example)

            # Add typos (10% of examples)
            if random.random() < 0.1:
                for correct, typo_list in self.typos.items():
                    if correct in example:
                        typo = random.choice(typo_list)
                        new_example = example.replace(correct, typo)
                        if new_example not in augmented:
                            augmented.append(new_example)

        return augmented


# ============ VALIDATOR ============

class DataValidator:
    """Validate training data quality"""

    def __init__(self):
        self.min_examples_per_intent = 10
        self.max_examples_per_intent = 500

    def validate_nlu_file(self, file_path: str) -> Dict:
        """Validate NLU file and return report"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        report = {
            'total_intents': 0,
            'total_examples': 0,
            'warnings': [],
            'errors': []
        }

        if 'nlu' not in data:
            report['errors'].append("Invalid NLU file format")
            return report

        intent_examples_count = {}

        for intent_data in data['nlu']:
            intent_name = intent_data.get('intent', 'unknown')
            examples_text = intent_data.get('examples', '')
            examples = [e.strip() for e in examples_text.split('\n') if e.strip().startswith('-')]

            num_examples = len(examples)
            intent_examples_count[intent_name] = num_examples
            report['total_examples'] += num_examples

            # Check minimum examples
            if num_examples < self.min_examples_per_intent:
                report['warnings'].append(
                    f"Intent '{intent_name}' has only {num_examples} examples (min: {self.min_examples_per_intent})"
                )

            # Check maximum examples
            if num_examples > self.max_examples_per_intent:
                report['warnings'].append(
                    f"Intent '{intent_name}' has {num_examples} examples (max: {self.max_examples_per_intent})"
                )

            # Check for duplicates
            unique_examples = set(examples)
            if len(unique_examples) < len(examples):
                duplicates = len(examples) - len(unique_examples)
                report['warnings'].append(
                    f"Intent '{intent_name}' has {duplicates} duplicate examples"
                )

        report['total_intents'] = len(intent_examples_count)

        return report

    def print_report(self, report: Dict):
        """Print validation report"""
        print("\n" + "=" * 60)
        print("📊 TRAINING DATA VALIDATION REPORT")
        print("=" * 60)
        print(f"\n✓ Total Intents: {report['total_intents']}")
        print(f"✓ Total Examples: {report['total_examples']}")
        print(f"✓ Average Examples per Intent: {report['total_examples'] / report['total_intents']:.1f}")

        if report['errors']:
            print(f"\n❌ ERRORS ({len(report['errors'])}):")
            for error in report['errors']:
                print(f"   • {error}")

        if report['warnings']:
            print(f"\n⚠️  WARNINGS ({len(report['warnings'])}):")
            for warning in report['warnings']:
                print(f"   • {warning}")

        if not report['errors'] and not report['warnings']:
            print("\n✅ All checks passed!")

        print("=" * 60 + "\n")


# ============ MAIN ============

def main():
    """Main function with examples"""
    print("=" * 60)
    print("🤖 RASA TRAINING DATA GENERATOR")
    print("=" * 60)

    # Example 1: Generate new NLU data
    print("\n1️⃣  Generating NLU data...")
    generator = TrainingDataGenerator()
    generator.generate_nlu_file('generated_nlu.yml')

    # Example 2: Annotate entities
    print("\n2️⃣  Annotating entities...")
    annotator = EntityAnnotator()
    # annotator.annotate_file('input.yml', 'output_annotated.yml')

    # Example 3: Validate data
    print("\n3️⃣  Validating data...")
    validator = DataValidator()
    report = validator.validate_nlu_file('generated_nlu.yml')
    validator.print_report(report)

    print("\n✅ Done! Check the generated files.")


if __name__ == '__main__':
    main()
