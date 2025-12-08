"""Unit tests for content processor models."""

from typing import Dict, Set

import pytest

from rasa.builder.copilot.models import (
    ReferenceEntry,
    ResponseCategory,
    ResponseCompleteness,
)
from rasa.builder.document_retrieval.models import Document
from rasa.builder.evaluator.content_processors.models import (
    CODE_EVIDENCE_TYPE,
    DOCUMENTATION_EVIDENCE_TYPE,
    Claim,
    ClaimImportance,
    Claims,
    CodeEvidence,
    DocumentationEvidence,
)
from rasa.builder.evaluator.dataset.models import (
    DatasetEntry,
    DatasetExpectedOutput,
    DatasetInput,
    DatasetMetadata,
    DatasetMetadataCopilotAdditionalContext,
)


class TestDocumentationEvidence:
    """Tests for DocumentationEvidence.from_dataset_entry() method."""

    @pytest.mark.parametrize(
        "dataset_entry, expected_used_count",
        [
            # Test case 1: All documents are used
            (
                DatasetEntry(
                    id="test-id-1",
                    input=DatasetInput(message="Test message"),
                    expected_output=DatasetExpectedOutput(
                        answer="Test answer",
                        response_category=ResponseCategory.COPILOT,
                        references=[
                            ReferenceEntry(
                                index=0,
                                title="Document 1",
                                url="https://example.com/doc1",
                                response_category=ResponseCategory.REFERENCE_ENTRY,
                                response_completeness=ResponseCompleteness.COMPLETE,
                            ),
                            ReferenceEntry(
                                index=1,
                                title="Document 2",
                                url="https://example.com/doc2",
                                response_category=ResponseCategory.REFERENCE_ENTRY,
                                response_completeness=ResponseCompleteness.COMPLETE,
                            ),
                        ],
                    ),
                    metadata=DatasetMetadata(
                        ids={"experiment_id": "exp-123"},
                        copilot_additional_context=(
                            DatasetMetadataCopilotAdditionalContext(
                                relevant_documents=[
                                    Document(
                                        content="This is document 1 content",
                                        url="https://example.com/doc1",
                                        title="Document 1",
                                    ),
                                    Document(
                                        content="This is document 2 content",
                                        url="https://example.com/doc2",
                                        title="Document 2",
                                    ),
                                ],
                                relevant_assistant_files={},
                            )
                        ),
                    ),
                ),
                2,
            ),
            # Test case 2: Some documents are used, some are not
            (
                DatasetEntry(
                    id="test-id-2",
                    input=DatasetInput(message="Test message"),
                    expected_output=DatasetExpectedOutput(
                        answer="Test answer",
                        response_category=ResponseCategory.COPILOT,
                        references=[
                            ReferenceEntry(
                                index=0,
                                title="Document 1",
                                url="https://example.com/doc1",
                                response_category=ResponseCategory.REFERENCE_ENTRY,
                                response_completeness=ResponseCompleteness.COMPLETE,
                            ),
                        ],
                    ),
                    metadata=DatasetMetadata(
                        ids={"experiment_id": "exp-123"},
                        copilot_additional_context=(
                            DatasetMetadataCopilotAdditionalContext(
                                relevant_documents=[
                                    Document(
                                        content="This is document 1 content",
                                        url="https://example.com/doc1",
                                        title="Document 1",
                                    ),
                                    Document(
                                        content="This is document 2 content",
                                        url="https://example.com/doc2",
                                        title="Document 2",
                                    ),
                                    Document(
                                        content="This is document 3 content",
                                        url="https://example.com/doc3",
                                        title="Document 3",
                                    ),
                                ],
                                relevant_assistant_files={},
                            )
                        ),
                    ),
                ),
                1,
            ),
            # Test case 3: No documents are used
            (
                DatasetEntry(
                    id="test-id-3",
                    input=DatasetInput(message="Test message"),
                    expected_output=DatasetExpectedOutput(
                        answer="Test answer",
                        response_category=ResponseCategory.COPILOT,
                        references=[],
                    ),
                    metadata=DatasetMetadata(
                        ids={"experiment_id": "exp-123"},
                        copilot_additional_context=(
                            DatasetMetadataCopilotAdditionalContext(
                                relevant_documents=[
                                    Document(
                                        content="This is document 1 content",
                                        url="https://example.com/doc1",
                                        title="Document 1",
                                    ),
                                    Document(
                                        content="This is document 2 content",
                                        url="https://example.com/doc2",
                                        title="Document 2",
                                    ),
                                ],
                                relevant_assistant_files={},
                            )
                        ),
                    ),
                ),
                0,
            ),
            # Test case 4: Empty documents list
            (
                DatasetEntry(
                    id="test-id-4",
                    input=DatasetInput(message="Test message"),
                    expected_output=DatasetExpectedOutput(
                        answer="Test answer",
                        response_category=ResponseCategory.COPILOT,
                        references=[],
                    ),
                    metadata=DatasetMetadata(
                        ids={"experiment_id": "exp-123"},
                        copilot_additional_context=(
                            DatasetMetadataCopilotAdditionalContext(
                                relevant_documents=[],
                                relevant_assistant_files={},
                            )
                        ),
                    ),
                ),
                0,
            ),
        ],
    )
    def test_from_dataset_entry_success(
        self,
        dataset_entry: DatasetEntry,
        expected_used_count: int,
    ) -> None:
        """Test successful creation of DocumentationEvidence from DatasetEntry.

        Args:
            dataset_entry: The dataset entry with documents and references.
            expected_used_count: Expected number of documents marked as used.
        """
        # When
        result = DocumentationEvidence.from_dataset_entry(dataset_entry)

        # Then
        relevant_documents = (
            dataset_entry.metadata.copilot_additional_context.relevant_documents
        )
        assert isinstance(result, list)
        assert len(result) == len(relevant_documents)

        # Verify all evidence items have correct structure
        for evidence in result:
            assert isinstance(evidence, DocumentationEvidence)
            assert evidence.type == DOCUMENTATION_EVIDENCE_TYPE
            assert evidence.url is not None
            assert evidence.content is not None
            assert isinstance(evidence.used, bool)
            assert isinstance(evidence.metadata, dict)

        # Verify the count of used documents
        used_count = sum(1 for evidence in result if evidence.used)
        assert used_count == expected_used_count

        # If there are documents, verify the content matches
        if relevant_documents:
            referenced_urls = {
                ref.url for ref in dataset_entry.expected_output.references
            }
            for i, evidence in enumerate(result):
                assert evidence.url == relevant_documents[i].url
                assert evidence.title == relevant_documents[i].title
                assert evidence.content == relevant_documents[i].content

                # Verify the used flag is correct
                expected_used = evidence.url in referenced_urls
                assert evidence.used == expected_used

    @pytest.mark.parametrize(
        "dataset_entry, error_message",
        [
            # Test case 1: Document with None URL
            (
                DatasetEntry(
                    id="test-id-fail-1",
                    input=DatasetInput(message="Test message"),
                    expected_output=DatasetExpectedOutput(
                        answer="Test answer",
                        response_category=ResponseCategory.COPILOT,
                        references=[],
                    ),
                    metadata=DatasetMetadata(
                        ids={"experiment_id": "exp-123"},
                        copilot_additional_context=(
                            DatasetMetadataCopilotAdditionalContext(
                                relevant_documents=[
                                    Document(
                                        content="This is document 1 content",
                                        url=None,
                                        title="Document 1",
                                    ),
                                ],
                                relevant_assistant_files={},
                            )
                        ),
                    ),
                ),
                "Document's URL is None",
            ),
            # Test case 2: Multiple documents, one with None URL
            (
                DatasetEntry(
                    id="test-id-fail-2",
                    input=DatasetInput(message="Test message"),
                    expected_output=DatasetExpectedOutput(
                        answer="Test answer",
                        response_category=ResponseCategory.COPILOT,
                        references=[],
                    ),
                    metadata=DatasetMetadata(
                        ids={"experiment_id": "exp-123"},
                        copilot_additional_context=(
                            DatasetMetadataCopilotAdditionalContext(
                                relevant_documents=[
                                    Document(
                                        content="This is document 1 content",
                                        url="https://example.com/doc1",
                                        title="Document 1",
                                    ),
                                    Document(
                                        content="This is document 2 content",
                                        url=None,
                                        title="Document 2",
                                    ),
                                ],
                                relevant_assistant_files={},
                            )
                        ),
                    ),
                ),
                "Document's URL is None",
            ),
        ],
    )
    def test_from_dataset_entry_failure_none_url(
        self,
        dataset_entry: DatasetEntry,
        error_message: str,
    ) -> None:
        """Test that documents with None URL are skipped (not included in results).

        Args:
            dataset_entry: Dataset entry with at least one document with None URL.
            error_message: Expected error message (not used, kept for compatibility).
        """
        # When
        result = DocumentationEvidence.from_dataset_entry(dataset_entry)

        # Then - documents with None URL should be skipped
        assert all(doc.url is not None for doc in result)


class TestCodeEvidence:
    """Tests for CodeEvidence.from_dataset_entry() method."""

    @pytest.mark.parametrize(
        "dataset_entry, expected_count",
        [
            # Test case 1: Multiple files
            (
                DatasetEntry(
                    id="test-id-1",
                    input=DatasetInput(message="Test message"),
                    expected_output=DatasetExpectedOutput(
                        answer="Test answer",
                        response_category=ResponseCategory.COPILOT,
                        references=[],
                    ),
                    metadata=DatasetMetadata(
                        ids={"experiment_id": "exp-123"},
                        copilot_additional_context=(
                            DatasetMetadataCopilotAdditionalContext(
                                relevant_documents=[],
                                relevant_assistant_files={
                                    "domain.yml": (
                                        "version: '3.1'\nintents:\n"
                                        "  - greet\n  - goodbye"
                                    ),
                                    "config.yml": (
                                        "language: en\npipeline:\n"
                                        "  - name: WhitespaceTokenizer"
                                    ),
                                    "actions/actions.py": (
                                        "from typing import Any\n\n"
                                        "class ActionHello:\n    pass"
                                    ),
                                },
                            )
                        ),
                    ),
                ),
                3,
            ),
            # Test case 2: Single file
            (
                DatasetEntry(
                    id="test-id-2",
                    input=DatasetInput(message="Test message"),
                    expected_output=DatasetExpectedOutput(
                        answer="Test answer",
                        response_category=ResponseCategory.COPILOT,
                        references=[],
                    ),
                    metadata=DatasetMetadata(
                        ids={"experiment_id": "exp-123"},
                        copilot_additional_context=(
                            DatasetMetadataCopilotAdditionalContext(
                                relevant_documents=[],
                                relevant_assistant_files={
                                    "domain.yml": "version: '3.1'\nintents:\n  - greet",
                                },
                            )
                        ),
                    ),
                ),
                1,
            ),
            # Test case 3: Empty files dictionary
            (
                DatasetEntry(
                    id="test-id-3",
                    input=DatasetInput(message="Test message"),
                    expected_output=DatasetExpectedOutput(
                        answer="Test answer",
                        response_category=ResponseCategory.COPILOT,
                        references=[],
                    ),
                    metadata=DatasetMetadata(
                        ids={"experiment_id": "exp-123"},
                        copilot_additional_context=(
                            DatasetMetadataCopilotAdditionalContext(
                                relevant_documents=[],
                                relevant_assistant_files={},
                            )
                        ),
                    ),
                ),
                0,
            ),
            # Test case 4: Files with empty content
            (
                DatasetEntry(
                    id="test-id-4",
                    input=DatasetInput(message="Test message"),
                    expected_output=DatasetExpectedOutput(
                        answer="Test answer",
                        response_category=ResponseCategory.COPILOT,
                        references=[],
                    ),
                    metadata=DatasetMetadata(
                        ids={"experiment_id": "exp-123"},
                        copilot_additional_context=(
                            DatasetMetadataCopilotAdditionalContext(
                                relevant_documents=[],
                                relevant_assistant_files={
                                    "domain.yml": "",
                                    "config.yml": "",
                                },
                            )
                        ),
                    ),
                ),
                2,
            ),
        ],
    )
    def test_from_dataset_entry_success(
        self,
        dataset_entry: DatasetEntry,
        expected_count: int,
    ) -> None:
        """Test successful creation of CodeEvidence from DatasetEntry.

        Args:
            dataset_entry: Dataset entry with relevant assistant files.
            expected_count: Expected number of CodeEvidence instances.
        """
        # Given: DatasetEntry with relevant assistant files

        # When: Create CodeEvidence from the dataset entry
        result = CodeEvidence.from_dataset_entry(dataset_entry)

        # Then: Verify the result
        relevant_assistant_files = (
            dataset_entry.metadata.copilot_additional_context.relevant_assistant_files
        )
        assert isinstance(result, list)
        assert len(result) == expected_count

        # Verify all evidence items have correct structure
        for evidence in result:
            assert isinstance(evidence, CodeEvidence)
            assert evidence.type == CODE_EVIDENCE_TYPE
            assert evidence.file_path is not None
            assert evidence.file_content is not None
            assert isinstance(evidence.referenced, bool)  # Should be computed
            assert isinstance(evidence.metadata, dict)

        # If there are files, verify the content matches
        if relevant_assistant_files:
            result_files = {
                evidence.file_path: evidence.file_content for evidence in result
            }
            assert result_files == relevant_assistant_files

    @pytest.mark.parametrize(
        "response_answer, assistant_files, expected_referenced",
        [
            # Test case 1: Files mentioned in response using **File: `path`** pattern
            (
                "Update **File: `domain.yml`** and **File: `config.yml`**",
                {
                    "domain.yml": "version: '3.1'",
                    "config.yml": "language: en",
                    "actions.py": "# Not referenced",
                },
                {
                    "domain.yml": True,
                    "config.yml": True,
                    "actions.py": False,
                },
            ),
            # Test case 2: Files mentioned with backticks and extensions
            (
                "Check `actions.py` and `endpoints.json` for configuration",
                {
                    "actions.py": "# Action code",
                    "endpoints.json": "{}",
                    "domain.yml": "# Not referenced",
                },
                {
                    "actions.py": True,
                    "endpoints.json": True,
                    "domain.yml": False,
                },
            ),
            # Test case 3: Files with directory paths
            (
                "See `data/flows/booking.yml` and `actions/actions.py`",
                {
                    "data/flows/booking.yml": "flows: []",
                    "actions/actions.py": "# Actions",
                    "config.yml": "# Not referenced",
                },
                {
                    "data/flows/booking.yml": True,
                    "actions/actions.py": True,
                    "config.yml": False,
                },
            ),
            # Test case 4: Partial filename matches
            (
                "Update `domain.yml`",
                {
                    "domain.yml": "# Domain",
                    "config/domain.yml": "# Config domain",
                    "actions.py": "# Not referenced",
                },
                {
                    "domain.yml": True,
                    "config/domain.yml": True,  # Filename matches
                    "actions.py": False,
                },
            ),
            # Test case 5: No files referenced
            (
                "This response doesn't mention any files",
                {
                    "domain.yml": "# Domain",
                    "config.yml": "# Config",
                },
                {
                    "domain.yml": False,
                    "config.yml": False,
                },
            ),
            # Test case 6: All files referenced
            (
                "Check **File: `domain.yml`**, `config.yml`, and `actions.py`",
                {
                    "domain.yml": "# Domain",
                    "config.yml": "# Config",
                    "actions.py": "# Actions",
                },
                {
                    "domain.yml": True,
                    "config.yml": True,
                    "actions.py": True,
                },
            ),
        ],
    )
    def test_from_dataset_entry_referenced_field(
        self,
        response_answer: str,
        assistant_files: Dict[str, str],
        expected_referenced: Dict[str, bool],
    ) -> None:
        """Test that referenced field is correctly computed based on response content.

        Args:
            response_answer: The Copilot response text.
            assistant_files: Dictionary of assistant files.
            expected_referenced: Expected referenced status for each file.
        """
        # Given: DatasetEntry with response and assistant files
        dataset_entry = DatasetEntry(
            id="test-id",
            input=DatasetInput(message="Test message"),
            expected_output=DatasetExpectedOutput(
                answer=response_answer,
                response_category=ResponseCategory.COPILOT,
                references=[],
            ),
            metadata=DatasetMetadata(
                ids={"experiment_id": "exp-123"},
                copilot_additional_context=DatasetMetadataCopilotAdditionalContext(
                    relevant_documents=[],
                    relevant_assistant_files=assistant_files,
                ),
            ),
        )

        # When: Create CodeEvidence from the dataset entry
        result = CodeEvidence.from_dataset_entry(dataset_entry)

        # Then: Verify referenced field is correctly set
        assert len(result) == len(assistant_files)
        result_dict = {evidence.file_path: evidence.referenced for evidence in result}
        assert result_dict == expected_referenced

    @pytest.mark.parametrize(
        "response_answer, assistant_files, use_only_referenced, expected_file_paths",
        [
            # Test case 1: Filter enabled - only referenced files
            (
                "Update **File: `domain.yml`** and `config.yml`",
                {
                    "domain.yml": "version: '3.1'",
                    "config.yml": "language: en",
                    "actions.py": "# Not referenced",
                },
                True,
                {"domain.yml", "config.yml"},
            ),
            # Test case 2: Filter disabled - all files included
            (
                "Update **File: `domain.yml`** and `config.yml`",
                {
                    "domain.yml": "version: '3.1'",
                    "config.yml": "language: en",
                    "actions.py": "# Not referenced",
                },
                False,
                {"domain.yml", "config.yml", "actions.py"},
            ),
            # Test case 3: Filter enabled - no files referenced
            (
                "This response doesn't mention any files",
                {
                    "domain.yml": "version: '3.1'",
                    "config.yml": "language: en",
                },
                True,
                set(),
            ),
            # Test case 4: Filter disabled - no files referenced but all returned
            (
                "This response doesn't mention any files",
                {
                    "domain.yml": "version: '3.1'",
                    "config.yml": "language: en",
                },
                False,
                {"domain.yml", "config.yml"},
            ),
            # Test case 5: Filter enabled - all files referenced
            (
                "Check `domain.yml`, `config.yml`, and `actions.py`",
                {
                    "domain.yml": "version: '3.1'",
                    "config.yml": "language: en",
                    "actions.py": "# Actions",
                },
                True,
                {"domain.yml", "config.yml", "actions.py"},
            ),
            # Test case 6: Filter enabled - partial match with directory paths
            (
                "See `data/flows/booking.yml`",
                {
                    "booking.yml": "flows: []",
                    "data/flows/booking.yml": "flows: []",
                    "config.yml": "# Not referenced",
                },
                True,
                {"booking.yml", "data/flows/booking.yml"},  # Both match
            ),
        ],
    )
    def test_from_dataset_entry_with_filtering(
        self,
        response_answer: str,
        assistant_files: Dict[str, str],
        use_only_referenced: bool,
        expected_file_paths: Set[str],
    ) -> None:
        """Test filtering behavior with use_only_files_referenced_in_response flag.

        Args:
            response_answer: The Copilot response text.
            assistant_files: Dictionary of assistant files.
            use_only_referenced: Whether to filter to only referenced files.
            expected_file_paths: Expected file paths in the result.
        """
        # Given: DatasetEntry with response and assistant files
        dataset_entry = DatasetEntry(
            id="test-id",
            input=DatasetInput(message="Test message"),
            expected_output=DatasetExpectedOutput(
                answer=response_answer,
                response_category=ResponseCategory.COPILOT,
                references=[],
            ),
            metadata=DatasetMetadata(
                ids={"experiment_id": "exp-123"},
                copilot_additional_context=DatasetMetadataCopilotAdditionalContext(
                    relevant_documents=[],
                    relevant_assistant_files=assistant_files,
                ),
            ),
        )

        # When: Create CodeEvidence with filtering flag
        result = CodeEvidence.from_dataset_entry(
            dataset_entry,
            use_only_files_referenced_in_response=use_only_referenced,
        )

        # Then: Verify correct files are returned
        result_file_paths = {evidence.file_path for evidence in result}
        assert result_file_paths == expected_file_paths

        # When filtering is enabled, all returned files should be referenced
        if use_only_referenced and result:
            for evidence in result:
                assert evidence.referenced is True

    @pytest.mark.parametrize(
        "response, expected_file_paths, referenced_file, expected_match",
        [
            # Test case 1: Main pattern **File: `path`** with exact match
            (
                """
                Here's the configuration:
                **File: `domain.yml`**
                And another file:
                **File: `data/flows/booking.yml`**
                """,
                {"domain.yml", "data/flows/booking.yml"},
                "domain.yml",
                True,
            ),
            # Test case 2: File paths with various extensions (regression test for bug)
            (
                """
                You can check these files:
                - `config.yml` for configuration
                - `actions.py` for custom actions
                - `endpoints.json` for endpoints
                - `README.md` for documentation
                - `data.txt` for data
                """,
                {"config.yml", "actions.py", "endpoints.json", "README.md", "data.txt"},
                "actions.py",
                True,
            ),
            # Test case 3: File paths with directory paths
            (
                """
                The files are located at:
                `actions/actions.py`
                `data/flows/booking.yml`
                `config/domain.yml`
                """,
                {"actions/actions.py", "data/flows/booking.yml", "config/domain.yml"},
                "actions.py",
                True,
            ),
            # Test case 4: Special Rasa file names
            (
                """
                Update these files:
                `domain.yml`
                `config.yml`
                `flows.yml`
                `actions.yml`
                `endpoints.yml`
                """,
                {
                    "domain.yml",
                    "config.yml",
                    "flows.yml",
                    "actions.yml",
                    "endpoints.yml",
                },
                "endpoints.yml",
                True,
            ),
            # Test case 5: Mixed patterns with no match
            (
                """
                **File: `main.yml`**
                Also check `config.py` and `test/domain.yml`.
                The `endpoints.json` file contains endpoint configuration.
                """,
                {"main.yml", "config.py", "test/domain.yml", "endpoints.json"},
                "actions.py",
                False,
            ),
            # Test case 6: Empty response
            ("", set(), "domain.yml", False),
            # Test case 7: No file path matches in response
            (
                "This is a response without any file paths mentioned.",
                set(),
                "domain.yml",
                False,
            ),
            # Test case 8: Duplicate paths (set deduplication)
            (
                """
                **File: `domain.yml`**
                Check `domain.yml` for configuration.
                Also see `domain.yml` in the root.
                """,
                {"domain.yml"},
                "domain.yml",
                True,
            ),
            # Test case 9: Multiple file extensions (ensures no tuples returned)
            (
                """
                Check these files:
                `test.py`
                `config.yml`
                `data.yaml`
                `settings.json`
                `readme.md`
                `notes.txt`
                """,
                {
                    "test.py",
                    "config.yml",
                    "data.yaml",
                    "settings.json",
                    "readme.md",
                    "notes.txt",
                },
                "config.yml",
                True,
            ),
            # Test case 10: Filename match with different paths
            (
                """
                Check `domain.yml` and `config.yml`.
                """,
                {"domain.yml", "config.yml"},
                "actions/domain.yml",
                True,
            ),
            # Test case 11: Referenced file as substring of response path
            (
                """
                Check `config/domain.yml` file.
                """,
                {"config/domain.yml"},
                "domain.yml",
                True,
            ),
            # Test case 12: Response path as substring of referenced file
            (
                """
                Update `flows/domain.yml`.
                """,
                {"flows/domain.yml"},
                "data/flows/domain.yml",
                True,
            ),
            # Test case 13: Partial filename mismatch
            (
                """
                Check `domain_old.yml` and `config.yml`.
                """,
                {"domain_old.yml", "config.yml"},
                "domain.yml",
                False,
            ),
        ],
    )
    def test_extract_and_check_file_paths(
        self,
        response: str,
        expected_file_paths: Set[str],
        referenced_file: str,
        expected_match: bool,
    ) -> None:
        """Test extraction and matching of file paths (integration test).

        This test verifies the complete workflow:
        1. Extract file paths from Copilot response
        2. Check if a referenced file is mentioned in extracted paths

        Args:
            response: The Copilot response text.
            expected_file_paths: Expected set of file paths to extract.
            referenced_file: The file path to check for usage.
            expected_match: Whether the referenced file should match.
        """
        # Given: Response text with file paths

        # When: Extract file paths from response
        extracted_paths = CodeEvidence._extract_file_paths_from_response(response)

        # Then: Verify correct extraction
        assert isinstance(extracted_paths, set)
        assert extracted_paths == expected_file_paths

        # Ensure no tuples are in the result (regression test for the bug we fixed)
        for item in extracted_paths:
            assert isinstance(item, str), f"Expected str, got {type(item)}: {item}"

        # When: Check if referenced file is used
        is_used = CodeEvidence._check_if_path_is_used(referenced_file, extracted_paths)

        # Then: Verify the match result
        assert is_used is expected_match


class TestClaims:
    """Tests for Claims model magic methods."""

    def test_len(self) -> None:
        """Test that __len__ returns the correct number of claims."""
        # Given
        claim1 = Claim(importance=ClaimImportance.HIGH, text="First claim")
        claim2 = Claim(importance=ClaimImportance.MEDIUM, text="Second claim")
        claim3 = Claim(importance=ClaimImportance.LOW, text="Third claim")

        # When
        claims = Claims(claims=[claim1, claim2, claim3])

        # Then
        assert len(claims) == 3

    def test_len_empty(self) -> None:
        """Test that __len__ returns 0 for empty claims."""
        # Given
        claims = Claims(claims=[])

        # When / Then
        assert len(claims) == 0

    def test_getitem(self) -> None:
        """Test that __getitem__ allows indexing into claims."""
        # Given
        claim1 = Claim(importance=ClaimImportance.HIGH, text="First claim")
        claim2 = Claim(importance=ClaimImportance.MEDIUM, text="Second claim")
        claim3 = Claim(importance=ClaimImportance.LOW, text="Third claim")
        claims = Claims(claims=[claim1, claim2, claim3])

        # When / Then
        assert claims[0] == claim1
        assert claims[1] == claim2
        assert claims[2] == claim3

    def test_getitem_negative_index(self) -> None:
        """Test that __getitem__ supports negative indexing."""
        # Given
        claim1 = Claim(importance=ClaimImportance.HIGH, text="First claim")
        claim2 = Claim(importance=ClaimImportance.MEDIUM, text="Second claim")
        claim3 = Claim(importance=ClaimImportance.LOW, text="Third claim")
        claims = Claims(claims=[claim1, claim2, claim3])

        # When / Then
        assert claims[-1] == claim3
        assert claims[-2] == claim2
        assert claims[-3] == claim1

    def test_getitem_index_error(self) -> None:
        """Test that __getitem__ raises IndexError for invalid index."""
        # Given
        claim1 = Claim(importance=ClaimImportance.HIGH, text="First claim")
        claims = Claims(claims=[claim1])

        # When / Then
        with pytest.raises(IndexError):
            _ = claims[10]
