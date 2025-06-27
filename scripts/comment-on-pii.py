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
GITHUB_API_VERSION = 'application/vnd.github.v3+json'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GitHubAPI:
    """GitHub API client for creating PR comments"""
    
    def __init__(self, github_token: str, repo_owner: str, repo_name: str):
        self.github_token = github_token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.request_headers = {
            'Authorization': f'token {github_token}',
            'Accept': GITHUB_API_VERSION
        }
        self.base_api_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}'
    
    def get_pr_files(self, pr_number: int) -> List[dict]:
        """Get list of files changed in pull request"""
        files_endpoint_url = f'{self.base_api_url}/pulls/{pr_number}/files'
        
        try:
            response = requests.get(
                files_endpoint_url, 
                headers=self.request_headers,
                timeout=REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            changed_files = response.json()
            logger.info(f"Retrieved {len(changed_files)} changed files from PR #{pr_number}")
            return changed_files
        except requests.RequestException as request_error:
            logger.error(f"Failed to get PR files: {request_error}")
            raise
    
    def create_line_review_comment(
        self,
        pr_number: int,
        commit_sha: str, 
        file_path: str,
        line_number: int,
        comment_body: str
    ) -> bool:
        """Create a review comment on a specific line of code"""
        review_comments_endpoint_url = f'{self.base_api_url}/pulls/{pr_number}/comments'
        comment_data = {
            'body': comment_body,
            'commit_id': commit_sha,
            'path': file_path,
            'line': line_number
        }
        
        try:
            response = requests.post(
                review_comments_endpoint_url, 
                headers=self.request_headers, 
                json=comment_data,
                timeout=REQUEST_TIMEOUT_SECONDS
            )
            
            if response.status_code == HTTP_SUCCESS_CREATED:
                logger.info(f"Successfully created review comment on {file_path}:{line_number}")
                return True
            else:
                logger.error(f"Failed to comment on {file_path}:{line_number} - Status: {response.status_code}, Response: {response.text}")
                return False
                
        except requests.RequestException as request_error:
            logger.error(f"Request failed when commenting on {file_path}:{line_number}: {request_error}")
            return False
    
    def create_general_pr_comment(self, pr_number: int, comment_body: str) -> bool:
        """Create a general comment on the pull request"""
        issue_comments_endpoint_url = f'{self.base_api_url}/issues/{pr_number}/comments'
        comment_data = {'body': comment_body}
        
        try:
            response = requests.post(
                issue_comments_endpoint_url, 
                headers=self.request_headers, 
                json=comment_data,
                timeout=REQUEST_TIMEOUT_SECONDS
            )
            
            if response.status_code == HTTP_SUCCESS_CREATED:
                logger.info(f"Successfully created general comment on PR #{pr_number}")
                return True
            else:
                logger.error(f"Failed to create summary comment - Status: {response.status_code}, Response: {response.text}")
                return False
                
        except requests.RequestException as request_error:
            logger.error(f"Request failed when creating summary comment: {request_error}")
            return False


def parse_added_lines_from_diff(pr_files: List[dict]) -> Dict[str, Set[int]]:
    """Parse pull request files to identify which lines were added"""
    files_with_added_lines = {}
    
    for pr_file in pr_files:
        if 'patch' not in pr_file:
            continue
            
        file_path = pr_file['filename']
        files_with_added_lines[file_path] = set()
        
        diff_lines = pr_file['patch'].split('\n')
        current_line_number = 0
        
        for diff_line in diff_lines:
            if diff_line.startswith('@@'):
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                hunk_header_match = re.search(r'@@ -\d+,?\d* \+(\d+),?\d* @@', diff_line)
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


def main():
    """Main function to process Semgrep results and comment on PR"""
    
    # Get required environment variables
    github_token = os.environ.get('GITHUB_TOKEN')
    pr_number = int(os.environ.get('PR_NUMBER'))
    repo_owner = os.environ.get('REPO_OWNER')
    repo_name = os.environ.get('REPO_NAME')
    commit_sha = os.environ.get('COMMIT_SHA')

    try:
        # Check if Semgrep results file exists
        semgrep_results_file = Path('semgrep-results.json')
        if not semgrep_results_file.exists():
            logger.info('No semgrep-results.json found')
            return
        
        # Load Semgrep results
        with open(semgrep_results_file, 'r') as results_file:
            semgrep_data = json.load(results_file)
        semgrep_findings = semgrep_data.get('results', [])
        
        logger.info(f'Found {len(semgrep_findings)} total Semgrep findings')
        
        # Initialize GitHub API client
        github_api_client = GitHubAPI(github_token, repo_owner, repo_name)
        
        # Handle case with no findings
        if len(semgrep_findings) == 0:
            no_issues_message = '## 🔍 Semgrep PII Detection Summary\n\n✅ No PII issues detected!'
            success = github_api_client.create_general_pr_comment(pr_number, no_issues_message)
            if success:
                logger.info("Posted 'no issues detected' summary comment")
            return
        
        # Get PR files and parse added lines
        pr_files = github_api_client.get_pr_files(pr_number)
        files_with_added_lines = parse_added_lines_from_diff(pr_files)
        
        # Process each Semgrep finding and group by line
        comments_posted_count = 0
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
                    # Collect ALL findings for this line
                    all_findings_for_line = [
                        f for f in semgrep_findings 
                        if f['path'] == finding_file_path and f['start']['line'] == finding_start_line
                        and any(line_num in files_with_added_lines.get(f['path'], set()) 
                               for line_num in range(f['start']['line'], f['end']['line'] + 1))
                    ]
                    
                    if len(all_findings_for_line) == 1:
                        # Single finding - use original format
                        pii_detection_comment = (
                            f"🔍 **PII Detection**: {semgrep_finding['extra']['message']}\n\n"
                            f"**Rule**: {semgrep_finding['check_id']}\n\n"
                            f"⚠️ Please review this potential PII exposure."
                        )
                    else:
                        # Multiple findings - combine them
                        rules_list = [f"• **{f['check_id']}**: {f['extra']['message']}" for f in all_findings_for_line]
                        pii_detection_comment = f"""🔍 **PII Detection**: Multiple issues detected:
{chr(10).join(rules_list)}

⚠️ Please review this potential PII exposure."""
                    
                    comment_creation_success = github_api_client.create_line_review_comment(
                        pr_number, commit_sha, finding_file_path, finding_start_line, pii_detection_comment
                    )
                    
                    if comment_creation_success:
                        comments_posted_count += 1
                    
                    commented_lines.add(comment_key)
                    
                    # Rate limiting delay
                    time.sleep(RATE_LIMIT_DELAY_SECONDS)
            else:
                findings_skipped_count += 1
        
        # Create summary comment
        if comments_posted_count > 0:
            summary_status_message = '⚠️ Individual line comments posted above.'
        else:
            summary_status_message = '✅ No PII issues in added lines!'
        
        summary_comment_text = f"""## 🔍 Semgrep PII Detection Summary

**Total findings:** {len(semgrep_findings)}
**Comments posted:** {comments_posted_count} (added lines)
**Skipped:** {findings_skipped_count} (existing lines)

{summary_status_message}"""
        
        summary_creation_success = github_api_client.create_general_pr_comment(
            pr_number, 
            summary_comment_text
        )
        
        if summary_creation_success:
            logger.info(f"Posted summary: {comments_posted_count} comments posted, {findings_skipped_count} findings skipped")
        
    except Exception as unexpected_error:
        logger.error(f"Error in PII detection workflow: {unexpected_error}")
        
        # Try to post error comment
        try:
            error_github_client = GitHubAPI(github_token, repo_owner, repo_name)
            error_message = '## 🔍 Semgrep PII Detection Summary\n\n❌ Error processing results - check workflow logs.'
            error_github_client.create_general_pr_comment(pr_number, error_message)
        except Exception:
            pass  # If we can't post error comment, just exit
        
        sys.exit(1)


if __name__ == '__main__':
    main()
