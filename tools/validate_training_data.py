#!/usr/bin/env python3
"""
Validate Rasa training data quality
Usage: python validate_training_data.py <path_to_nlu_file>
"""

import sys
import yaml
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import re


class TrainingDataValidator:
    """Comprehensive training data validation"""

    def __init__(self):
        self.warnings = []
        self.errors = []
        self.stats = {}

        # Thresholds
        self.MIN_EXAMPLES = 10
        self.RECOMMENDED_EXAMPLES = 20
        self.MAX_EXAMPLES = 500
        self.MIN_WORD_LENGTH = 2
        self.DUPLICATE_THRESHOLD = 0.1  # 10% duplicates is too much

    def validate_file(self, file_path: str) -> Dict:
        """Main validation function"""
        print(f"\n{'='*70}")
        print(f"🔍 VALIDATING: {file_path}")
        print(f"{'='*70}\n")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            self.errors.append(f"Failed to load file: {str(e)}")
            return self._generate_report()

        if 'nlu' not in data:
            self.errors.append("Invalid NLU file format - missing 'nlu' key")
            return self._generate_report()

        # Run all checks
        self._check_version(data)
        self._check_intents(data['nlu'])
        self._check_examples_quality(data['nlu'])
        self._check_entities(data['nlu'])
        self._check_intent_similarity(data['nlu'])

        return self._generate_report()

    def _check_version(self, data: Dict):
        """Check Rasa version"""
        if 'version' not in data:
            self.warnings.append("Missing 'version' field")
        elif data['version'] != '3.1':
            self.warnings.append(f"Version is {data['version']}, recommended: 3.1")

    def _check_intents(self, nlu_data: List[Dict]):
        """Check intents structure and count"""
        intent_examples = {}

        for item in nlu_data:
            if 'intent' not in item:
                # Might be synonym, regex, lookup
                continue

            intent_name = item['intent']
            examples_text = item.get('examples', '')
            examples = self._parse_examples(examples_text)

            intent_examples[intent_name] = examples

            # Check minimum examples
            num_examples = len(examples)
            if num_examples < self.MIN_EXAMPLES:
                self.errors.append(
                    f"❌ Intent '{intent_name}': Only {num_examples} examples "
                    f"(minimum: {self.MIN_EXAMPLES})"
                )
            elif num_examples < self.RECOMMENDED_EXAMPLES:
                self.warnings.append(
                    f"⚠️  Intent '{intent_name}': Only {num_examples} examples "
                    f"(recommended: {self.RECOMMENDED_EXAMPLES})"
                )

            # Check maximum examples
            if num_examples > self.MAX_EXAMPLES:
                self.warnings.append(
                    f"⚠️  Intent '{intent_name}': {num_examples} examples "
                    f"(might be too many, consider reviewing)"
                )

        self.stats['total_intents'] = len(intent_examples)
        self.stats['total_examples'] = sum(len(ex) for ex in intent_examples.values())
        self.stats['intent_examples'] = intent_examples

    def _check_examples_quality(self, nlu_data: List[Dict]):
        """Check quality of examples"""
        for item in nlu_data:
            if 'intent' not in item:
                continue

            intent_name = item['intent']
            examples = self.stats['intent_examples'].get(intent_name, [])

            if not examples:
                continue

            # Check for duplicates
            duplicates = len(examples) - len(set(examples))
            if duplicates > 0:
                dup_rate = duplicates / len(examples)
                if dup_rate > self.DUPLICATE_THRESHOLD:
                    self.errors.append(
                        f"❌ Intent '{intent_name}': {duplicates} duplicate examples "
                        f"({dup_rate*100:.1f}%)"
                    )
                else:
                    self.warnings.append(
                        f"⚠️  Intent '{intent_name}': {duplicates} duplicate examples"
                    )

            # Check for very short examples
            short_examples = [ex for ex in examples if len(ex.split()) < self.MIN_WORD_LENGTH]
            if short_examples:
                self.warnings.append(
                    f"⚠️  Intent '{intent_name}': {len(short_examples)} examples "
                    f"with < {self.MIN_WORD_LENGTH} words"
                )

            # Check for very similar examples (simple heuristic)
            similar_count = self._count_similar_examples(examples)
            if similar_count > len(examples) * 0.3:  # >30% similar
                self.warnings.append(
                    f"⚠️  Intent '{intent_name}': Many similar examples detected "
                    f"(diversity might be low)"
                )

            # Check variation in length
            lengths = [len(ex.split()) for ex in examples]
            if max(lengths) - min(lengths) < 3:
                self.warnings.append(
                    f"⚠️  Intent '{intent_name}': Low length variation "
                    f"(all examples have similar length)"
                )

    def _check_entities(self, nlu_data: List[Dict]):
        """Check entity annotations"""
        entity_counts = defaultdict(int)

        for item in nlu_data:
            if 'intent' not in item:
                continue

            intent_name = item['intent']
            examples = self.stats['intent_examples'].get(intent_name, [])

            for example in examples:
                # Find entity annotations [text](entity_type)
                entities = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', example)
                for _, entity_type in entities:
                    entity_counts[entity_type] += 1

        self.stats['entity_counts'] = dict(entity_counts)

        # Check if entities are used consistently
        for entity_type, count in entity_counts.items():
            if count < 5:
                self.warnings.append(
                    f"⚠️  Entity '{entity_type}': Only used {count} times "
                    f"(might need more examples)"
                )

    def _check_intent_similarity(self, nlu_data: List[Dict]):
        """Check for potentially overlapping intents"""
        intent_examples = self.stats.get('intent_examples', {})

        if len(intent_examples) < 2:
            return

        intent_names = list(intent_examples.keys())

        for i, intent1 in enumerate(intent_names):
            for intent2 in intent_names[i+1:]:
                examples1 = set(self._normalize_examples(intent_examples[intent1]))
                examples2 = set(self._normalize_examples(intent_examples[intent2]))

                overlap = examples1 & examples2
                if overlap:
                    self.errors.append(
                        f"❌ Intents '{intent1}' and '{intent2}' have {len(overlap)} "
                        f"identical examples!"
                    )

                # Check semantic similarity (simple word overlap)
                similarity = self._calculate_intent_similarity(examples1, examples2)
                if similarity > 0.5:
                    self.warnings.append(
                        f"⚠️  Intents '{intent1}' and '{intent2}' seem very similar "
                        f"(similarity: {similarity:.2%})"
                    )

    def _parse_examples(self, examples_text: str) -> List[str]:
        """Parse examples from YAML text"""
        examples = []
        for line in examples_text.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                # Remove entity annotations for counting
                example = line[2:].strip()
                examples.append(example)
        return examples

    def _normalize_examples(self, examples: List[str]) -> List[str]:
        """Normalize examples for comparison (remove entity annotations)"""
        normalized = []
        for ex in examples:
            # Remove entity annotations [text](type)
            clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', ex)
            clean = clean.lower().strip()
            normalized.append(clean)
        return normalized

    def _count_similar_examples(self, examples: List[str]) -> int:
        """Count potentially similar examples"""
        # Simple heuristic: count examples that start with same 2 words
        normalized = self._normalize_examples(examples)
        prefixes = []

        for ex in normalized:
            words = ex.split()
            if len(words) >= 2:
                prefix = ' '.join(words[:2])
                prefixes.append(prefix)

        # Count duplicates in prefixes
        prefix_counts = Counter(prefixes)
        similar = sum(count - 1 for count in prefix_counts.values() if count > 1)
        return similar

    def _calculate_intent_similarity(self, examples1: set, examples2: set) -> float:
        """Calculate similarity between two sets of examples"""
        if not examples1 or not examples2:
            return 0.0

        # Word-level Jaccard similarity
        words1 = set()
        words2 = set()

        for ex in examples1:
            words1.update(ex.split())
        for ex in examples2:
            words2.update(ex.split())

        intersection = words1 & words2
        union = words1 | words2

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _generate_report(self) -> Dict:
        """Generate validation report"""
        return {
            'stats': self.stats,
            'warnings': self.warnings,
            'errors': self.errors
        }

    def print_report(self, report: Dict):
        """Print beautiful report"""
        stats = report['stats']
        warnings = report['warnings']
        errors = report['errors']

        print(f"\n{'='*70}")
        print("📊 VALIDATION REPORT")
        print(f"{'='*70}\n")

        # Stats
        if stats:
            print("📈 STATISTICS:")
            print(f"   • Total Intents: {stats.get('total_intents', 0)}")
            print(f"   • Total Examples: {stats.get('total_examples', 0)}")

            if stats.get('total_intents', 0) > 0:
                avg = stats.get('total_examples', 0) / stats.get('total_intents', 1)
                print(f"   • Average Examples per Intent: {avg:.1f}")

            if stats.get('entity_counts'):
                print(f"\n   Entities Found:")
                for entity, count in sorted(stats['entity_counts'].items()):
                    print(f"      - {entity}: {count} occurrences")

            if stats.get('intent_examples'):
                print(f"\n   Examples per Intent:")
                for intent, examples in sorted(stats['intent_examples'].items()):
                    print(f"      - {intent}: {len(examples)} examples")

        # Errors
        if errors:
            print(f"\n{'='*70}")
            print(f"❌ ERRORS ({len(errors)}):")
            print(f"{'='*70}")
            for error in errors:
                print(f"   {error}")

        # Warnings
        if warnings:
            print(f"\n{'='*70}")
            print(f"⚠️  WARNINGS ({len(warnings)}):")
            print(f"{'='*70}")
            for warning in warnings:
                print(f"   {warning}")

        # Summary
        print(f"\n{'='*70}")
        if not errors and not warnings:
            print("✅ ALL CHECKS PASSED! Training data looks good.")
        elif errors:
            print(f"❌ FAILED: {len(errors)} error(s) found. Please fix before training.")
        else:
            print(f"⚠️  PASSED WITH WARNINGS: {len(warnings)} warning(s).")
            print("   Consider addressing warnings for better quality.")
        print(f"{'='*70}\n")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python validate_training_data.py <path_to_nlu_file>")
        print("\nExample:")
        print("  python validate_training_data.py data/nlu.yml")
        sys.exit(1)

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"❌ Error: File '{file_path}' not found")
        sys.exit(1)

    validator = TrainingDataValidator()
    report = validator.validate_file(file_path)
    validator.print_report(report)

    # Exit code
    if report['errors']:
        sys.exit(1)  # Fail with errors
    else:
        sys.exit(0)  # Success


if __name__ == '__main__':
    main()
