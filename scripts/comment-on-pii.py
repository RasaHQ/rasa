"""
Script to comment on PII findings in GitHub PR
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Set, List, Optional

import requests

# Constants
RATE_LIMIT_DELAY_SECONDS = 1
REQUEST_TIMEOUT_SECONDS = 30
HTTP_SUCCESS_CREATED = 201
HTTP_SUCCESS_OK = 200
GITHUB_API_VERSION = "application/vnd.github.v3+json"
PII_COMMENT_MARKER = "🔍 **PII Detection**"
SUMMARY_COMMENT_MARKER = "## 🔍 Semgrep PII Detection Summary"
REPO_URL = "https://api.github.com/repos"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GitHubAPI:
    """GitHub API client for creating PR comments."""
    
    def __init__(self, github_token: str, repo_owner: str, repo_name: str):
        self.github_token = github_token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.request_headers = {
            'Authorization': f'token {github_token}',
            'Accept': GITHUB_API_VERSION,
            'User-Agent': 'PII-Detection-Bot/1.0'
        }
        self.base_api_url = f'{REPO_URL}/{repo_owner}/{repo_name}'

    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make HTTP request with consistent error handling."""
        try:
            response = requests.request(
                method, 
                url, 
                headers=self.request_headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
                **kwargs
            )
            
            if not response.ok:
                logger.error(f"HTTP {response.status_code} error - URL: {url}, Response: {response.text}")
            
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Request failed - URL: {url}, Error: {e}")
            raise

    def get_pr_files(self, pr_number: int) -> List[dict]:
        """Get list of files changed in pull request."""
        files_endpoint_url = f'{self.base_api_url}/pulls/{pr_number}/files'
        
        response = self._make_request('GET', files_endpoint_url)
        changed_files = response.json()
        logger.info(f"Retrieved {len(changed_files)} changed files from PR #{pr_number}")
        return changed_files
    
    def get_existing_review_comments(self, pr_number: int) -> List[dict]:
        """Get existing review comments for the PR."""
        comments_endpoint_url = f'{self.base_api_url}/pulls/{pr_number}/comments'
        
        response = self._make_request('GET', comments_endpoint_url)
        comments = response.json()
        logger.info(f"Retrieved {len(comments)} existing review comments from PR #{pr_number}")
        return comments
    
    def get_existing_issue_comments(self, pr_number: int) -> List[dict]:
        """Get existing issue comments for the PR."""
        comments_endpoint_url = f'{self.base_api_url}/issues/{pr_number}/comments'
        
        response = self._make_request('GET', comments_endpoint_url)
        comments = response.json()
        logger.info(f"Retrieved {len(comments)} existing issue comments from PR #{pr_number}")
        return comments
    
    def create_line_review_comment(
        self,
        pr_number: int,
        commit_sha: str, 
        file_path: str,
        line_number: int,
        comment_body: str
    ) -> bool:
        """Create a review comment on a specific line of code."""
        review_comments_endpoint_url = f'{self.base_api_url}/pulls/{pr_number}/comments'
        comment_data = {
            'body': comment_body,
            'commit_id': commit_sha,
            'path': file_path,
            'line': line_number
        }
        
        try:
            self._make_request('POST', review_comments_endpoint_url, json=comment_data)
            logger.info(f"Successfully created review comment on {file_path}:{line_number}")
            return True
        except requests.RequestException:
            logger.error(f"Failed to comment on {file_path}:{line_number}")
            return False
    
    def update_review_comment(self, comment_id: int, comment_body: str) -> bool:
        """Update an existing review comment."""
        update_comment_url = f'{REPO_URL}/{self.repo_owner}/{self.repo_name}/pulls/comments/{comment_id}'
        comment_data = {'body': comment_body}
        
        try:
            self._make_request('PATCH', update_comment_url, json=comment_data)
            logger.info(f"Successfully updated review comment {comment_id}")
            return True
        except requests.RequestException:
            logger.error(f"Failed to update review comment {comment_id}")
            return False
    
    def create_general_pr_comment(self, pr_number: int, comment_body: str) -> bool:
        """Create a general comment on the pull request."""
        issue_comments_endpoint_url = f'{self.base_api_url}/issues/{pr_number}/comments'
        comment_data = {'body': comment_body}
        
        try:
            self._make_request('POST', issue_comments_endpoint_url, json=comment_data)
            logger.info(f"Successfully created general comment on PR #{pr_number}")
            return True
        except requests.RequestException:
            logger.error("Failed to create summary comment")
            return False

    def delete_general_pr_comment(self, comment_id: int) -> bool:
        """Delete an existing general comment on the pull request."""
        delete_comment_url = f'{REPO_URL}/{self.repo_owner}/{self.repo_name}/issues/comments/{comment_id}'
        
        try:
            self._make_request('DELETE', delete_comment_url)
            logger.info(f"Successfully deleted comment {comment_id}")
            return True
        except requests.RequestException:
            logger.error(f"Failed to delete comment {comment_id}")
            return False


def parse_added_lines_from_diff(pr_files: List[dict]) -> Dict[str, Set[int]]:
    """Parse pull request files to identify which lines were added."""
    files_with_added_lines = {}
    
    for pr_file in pr_files:
        # Skip files without patches or binary files
        if 'patch' not in pr_file or pr_file.get('binary', False):
            continue
            
        file_path = pr_file['filename']
        files_with_added_lines[file_path] = set()
        
        diff_lines = pr_file['patch'].split('\n')
        current_line_number = 0
        
        for diff_line in diff_lines:
            if diff_line.startswith('@@'):
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                hunk_header_match = re.search(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@', diff_line)
                if hunk_header_match:
                    current_line_number = int(hunk_header_match.group(1))
            elif diff_line.startswith('+') and not diff_line.startswith('+++'):
                # This is an added line (not a file header)
                files_with_added_lines[file_path].add(current_line_number)
                current_line_number += 1
            elif not diff_line.startswith('-') and not diff_line.startswith('\\'):
                # Context line (unchanged line)
                current_line_number += 1
    
    total_added_lines = sum(len(line_set) for line_set in files_with_added_lines.values())
    logger.info(f"Parsed {len(files_with_added_lines)} files with {total_added_lines} total added lines")
    
    return files_with_added_lines


def find_existing_pii_comment(existing_comments: List[dict], file_path: str, line_number: int) -> Optional[dict]:
    """Find existing PII comment for the given file and line."""
    for comment in existing_comments:
        # Check if this is a review comment (line-specific) and not outdated
        if (comment.get('path') == file_path and 
            not comment.get('outdated', False) and
            PII_COMMENT_MARKER in comment.get('body', '')):
            
            # Check if it's on the same line
            comment_line = comment.get('line') or comment.get('original_line')
            if comment_line == line_number:
                return comment
    return None


def should_update_comment(existing_comment_body: str, new_comment_body: str) -> bool:
    """Check if existing comment needs to be updated by comparing content."""
    # Normalize whitespace and line endings for comparison
    existing_normalized = re.sub(r'\s+', ' ', existing_comment_body.strip())
    new_normalized = re.sub(r'\s+', ' ', new_comment_body.strip())
    
    return existing_normalized != new_normalized


def find_existing_summary_comment(existing_comments: List[dict]) -> Optional[dict]:
    """Find existing Semgrep PII Detection Summary comment."""
    for comment in existing_comments:
        if SUMMARY_COMMENT_MARKER in comment.get('body', ''):
            return comment
    return None


def create_pii_comment_text(findings: List[dict]) -> str:
    """Create the comment text for PII findings."""
    if len(findings) == 1:
        finding = findings[0]
        return (
            f"{PII_COMMENT_MARKER}: {finding['extra']['message']}\n\n"
            f"**Rule**: `{finding['check_id']}`\n\n"
            f"⚠️ Please review this potential PII exposure and consider:\n"
            f"- Using debug-level logging for sensitive data\n"
            f"- Sanitizing or redacting PII before logging\n"
            f"- Moving detailed logging to secure audit logs"
        )
    else:
        # Multiple findings - combine them
        rules_list = [f"• **`{f['check_id']}`**: {f['extra']['message']}" for f in findings]
        return f"""{PII_COMMENT_MARKER}: Multiple issues detected:

{chr(10).join(rules_list)}

⚠️ Please review these potential PII exposures and consider:
- Using debug-level logging for sensitive data
- Sanitizing or redacting PII before logging
- Moving detailed logging to secure audit logs"""


def load_semgrep_results() -> List[dict]:
    """Load and parse Semgrep results from file."""
    semgrep_results_file = Path('semgrep-results.json')
    if not semgrep_results_file.exists():
        logger.info('No semgrep-results.json found')
        return []
    
    with open(semgrep_results_file, 'r') as results_file:
        semgrep_data = json.load(results_file)
    
    return semgrep_data.get('results', [])


def handle_no_findings(
    github_api_client: GitHubAPI,
    pr_number: int,
    existing_review_comments: List[dict],
    existing_issue_comments: List[dict]
) -> None:
    """Handle the case when no Semgrep findings are detected."""
    # Check if there are any existing PII comments from previous runs
    existing_pii_comments = [
        comment for comment in existing_review_comments
        if PII_COMMENT_MARKER in comment.get('body', '') and not comment.get('outdated', False) and not comment.get('resolved', False)
    ]
    
    if existing_pii_comments:
        logger.info(f"No new PII issues detected. {len(existing_pii_comments)} existing PII comment(s) from previous scans remain active. No new comment will be posted.")
    else:
        logger.info("No PII issues detected. No comment will be posted.")


def process_semgrep_findings(
    github_api_client: GitHubAPI,
    pr_number: int,
    commit_sha: str,
    semgrep_findings: List[dict],
    files_with_added_lines: Dict[str, Set[int]],
    existing_review_comments: List[dict]
) -> tuple:
    """Process Semgrep findings and create/update comments."""
    comments_posted_count = 0
    comments_updated_count = 0
    findings_skipped_count = 0
    commented_lines = set()
    
    for semgrep_finding in semgrep_findings:
        # Check if any line in the finding range was added
        finding_affects_added_lines = False
        finding_start_line = semgrep_finding['start']['line']
        finding_end_line = semgrep_finding['end']['line']
        finding_file_path = semgrep_finding['path']
        
        if finding_file_path in files_with_added_lines:
            for line_number_in_range in range(finding_start_line, finding_end_line + 1):
                if line_number_in_range in files_with_added_lines[finding_file_path]:
                    finding_affects_added_lines = True
                    break
        
        if finding_affects_added_lines:
            comment_key = (finding_file_path, finding_start_line)
            
            if comment_key not in commented_lines:
                # Collect all findings for this line
                all_findings_for_line = [
                    f for f in semgrep_findings 
                    if f['path'] == finding_file_path and f['start']['line'] == finding_start_line
                    and any(line_num in files_with_added_lines.get(f['path'], set()) 
                           for line_num in range(f['start']['line'], f['end']['line'] + 1))
                ]
                
                # Create comment text
                pii_detection_comment = create_pii_comment_text(all_findings_for_line)
                
                # Check if we already have a PII comment for this line
                existing_comment = find_existing_pii_comment(existing_review_comments, finding_file_path, finding_start_line)
                
                if existing_comment:
                    # Check if we need to update the existing comment
                    if should_update_comment(existing_comment['body'], pii_detection_comment):
                        logger.info(f"Updating existing PII comment on {finding_file_path}:{finding_start_line}")
                        if github_api_client.update_review_comment(existing_comment['id'], pii_detection_comment):
                            comments_updated_count += 1
                    else:
                        logger.info(f"Skipping {finding_file_path}:{finding_start_line} - PII comment is already up to date")
                        findings_skipped_count += len(all_findings_for_line)
                else:
                    # Create new comment
                    if github_api_client.create_line_review_comment(
                        pr_number, commit_sha, finding_file_path, finding_start_line, pii_detection_comment
                    ):
                        comments_posted_count += 1
                
                commented_lines.add(comment_key)
                
                # Rate limiting delay
                time.sleep(RATE_LIMIT_DELAY_SECONDS)
        else:
            findings_skipped_count += 1
    
    return comments_posted_count, comments_updated_count, findings_skipped_count


def create_and_post_summary(
    github_api_client: GitHubAPI,
    pr_number: int,
    semgrep_findings: List[dict],
    comments_posted_count: int,
    comments_updated_count: int,
    findings_skipped_count: int,
    existing_issue_comments: List[dict]
) -> None:
    """Create or update summary comment."""
    total_actions = comments_posted_count + comments_updated_count
    
    if len(semgrep_findings) == findings_skipped_count:
        logger.info("All findings were skipped (existing lines only). No summary comment will be posted.")
        return
    
    if total_actions == 0:
        logger.info("No actionable PII issues in added lines - exiting silently (no comment posted)")
        return
    
    if comments_posted_count > 0 and comments_updated_count > 0:
        summary_status_message = f'⚠️ {comments_posted_count} new comments posted, {comments_updated_count} existing comments updated.'
    elif comments_posted_count > 0:
        summary_status_message = f'⚠️ {comments_posted_count} new comments posted.'
    else:
        summary_status_message = f'⚠️ {comments_updated_count} existing comments updated.'

    summary_comment_text = f"""{SUMMARY_COMMENT_MARKER}

**Total findings:** {len(semgrep_findings)}
**New comments posted:** {comments_posted_count}
**Comments updated:** {comments_updated_count}
**Skipped:** {findings_skipped_count} (existing lines or unchanged comments)

{summary_status_message}

---
*💡 **Tip**: To reduce PII exposure risks, consider using debug-level logging for sensitive data or implementing data sanitization before logging.*"""
    
    # Check if summary comment already exists
    existing_summary = find_existing_summary_comment(existing_issue_comments)
    if existing_summary:
        if should_update_comment(existing_summary['body'], summary_comment_text):
            # Delete old summary and create new one
            if github_api_client.delete_general_pr_comment(existing_summary['id']):
                summary_creation_success = github_api_client.create_general_pr_comment(pr_number, summary_comment_text)
                if summary_creation_success:
                    logger.info("Replaced existing summary comment")
        else:
            logger.info("Skipping summary update - content unchanged")
            summary_creation_success = True  # Consider it successful since we intentionally skipped
    else:
        summary_creation_success = github_api_client.create_general_pr_comment(pr_number, summary_comment_text)
    
    if summary_creation_success:
        action_summary = []
        if comments_posted_count > 0:
            action_summary.append(f"{comments_posted_count} new comments")
        if comments_updated_count > 0:
            action_summary.append(f"{comments_updated_count} updated comments")
        if findings_skipped_count > 0:
            action_summary.append(f"{findings_skipped_count} skipped findings")
        
        logger.info(f"Summary updated: {', '.join(action_summary) if action_summary else 'no actions taken'}")


def main():
    """Main function to process Semgrep results and comment on PR."""
    
    # Get required environment variables
    github_token = os.environ.get('GITHUB_TOKEN')
    pr_number = int(os.environ.get('PR_NUMBER'))
    repo_owner = os.environ.get('REPO_OWNER')
    repo_name = os.environ.get('REPO_NAME')
    commit_sha = os.environ.get('COMMIT_SHA')

    try:
        # Load Semgrep results
        semgrep_findings = load_semgrep_results()
        if not semgrep_findings and not Path('semgrep-results.json').exists():
            return
        
        logger.info(f'Found {len(semgrep_findings)} total Semgrep findings')
        
        # Initialize GitHub API client
        github_api_client = GitHubAPI(github_token, repo_owner, repo_name)
        
        # Get existing comments
        existing_review_comments = github_api_client.get_existing_review_comments(pr_number)
        existing_issue_comments = github_api_client.get_existing_issue_comments(pr_number)
        
        # Handle case with no findings
        if len(semgrep_findings) == 0:
            handle_no_findings(github_api_client, pr_number, existing_review_comments, existing_issue_comments)
            return
        
        # Get PR files and parse added lines
        pr_files = github_api_client.get_pr_files(pr_number)
        files_with_added_lines = parse_added_lines_from_diff(pr_files)
        
        # Process findings and create/update comments
        comments_posted_count, comments_updated_count, findings_skipped_count = process_semgrep_findings(
            github_api_client,
            pr_number,
            commit_sha,
            semgrep_findings,
            files_with_added_lines,
            existing_review_comments
        )
        
        # Create or update summary comment
        create_and_post_summary(
            github_api_client,
            pr_number,
            semgrep_findings,
            comments_posted_count,
            comments_updated_count,
            findings_skipped_count,
            existing_issue_comments
        )
        
    except Exception as unexpected_error:
        logger.error(f"Error in PII detection workflow: {unexpected_error}")
        
        # Try to post error comment
        try:
            error_github_client = GitHubAPI(github_token, repo_owner, repo_name)
            error_message = f"{SUMMARY_COMMENT_MARKER}\n\n❌ Error processing results - check workflow logs.\n\nError: `{str(unexpected_error)}`"
            error_github_client.create_general_pr_comment(pr_number, error_message)
        except Exception:
            pass  # If we can't post error comment, just exit
        
        sys.exit(1)


if __name__ == '__main__':
    main()
