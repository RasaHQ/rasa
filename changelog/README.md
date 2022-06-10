This directory contains "newsfragments" which are short files that contain a small
**Markdown**-formatted text that will be added to the next `CHANGELOG`.

The `CHANGELOG` will be read by **users**, so this description should be aimed
to Rasa OSS users.

Make sure to use full sentences in the **past or present tense** and use
punctuation, examples:

    Slots will be correctly interpolated if there are lists in custom response templates.

    Previously this resulted in no interpolation.

Each file should be named like `<ISSUE>.<TYPE>.md`, where
`<ISSUE>` is an issue / PR number, and `<TYPE>` is one of:

* `feature`: new user facing features, like new command-line options and new behavior.
* `improvement`: improvement of existing functionality, usually without requiring user intervention.
* `bugfix`: fixes a reported bug or security vulnerability.
* `doc`: documentation improvement, like rewording an entire section or adding missing docs.
* `removal`: feature deprecation or feature removal.
* `misc`: fixing a small typo or internal change, will not be included in the changelog.

So for example: `123.feature.md`, `456.bugfix.md`.

If your change fixes an issue, use the issue number here. If there is no issue,
then after you submit the PR with your changes and get the PR number, you can add a
changelog using that PR number instead.

If you are not sure what issue type to use, don't hesitate to ask in your PR.

`towncrier` preserves multiple paragraphs and formatting (code blocks, lists,
and so on), but for entries other than `features` it is usually better to stick
to a single paragraph to keep it concise. You can install `towncrier` and then
run `towncrier --draft` if you want to get a preview of how your change will look
in the final release notes.
