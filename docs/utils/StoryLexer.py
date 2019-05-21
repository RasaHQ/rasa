from pygments.lexer import RegexLexer, bygroups, using, default, include
from pygments.lexers.data import JsonLexer
from pygments.token import Keyword, Comment, Token, Text, Generic, Name


class StoryLexer(RegexLexer):
    """Lexer for the Rasa Core story file format.
    Used for syntax highlighting of story snippets in the docs."""

    name = "Story"
    aliases = ["story"]
    filenames = ["*.md"]

    tokens = {
        "comment": [
            (
                r"(\s*<!--)((?:.*?\n?)*)(-->)",
                bygroups(Keyword, Comment.MultiLine, Keyword),
            )
        ],
        "root": [
            include("comment"),
            (r"\s*-\s*(slot)", Token.Operator, ("event", "event_rx")),
            (r"\s*-\s*(restart)", Token.Operator, ("event", "event_rx")),
            (r"\s*-\s*(rewind)", Token.Operator, ("event", "event_rx")),
            (r"\s*-\s*(reset_slots)", Token.Operator, ("event", "event_rx")),
            (r"\s*-\s*(reminder)", Token.Operator, ("event", "event_rx")),
            (r"\s*-\s*(undo)", Token.Operator, ("event", "event_rx")),
            (r"\s*-\s*(export)", Token.Operator, ("event", "event_rx")),
            (r"\s*-\s*(pause)", Token.Operator, ("event", "event_rx")),
            (r"\s*-\s*(resume)", Token.Operator, ("event", "event_rx")),
            (r"\s*-\s*(utter_[^\s]*)", Token.Text, ("event", "event_rx")),
            (
                r"(\s*-(?:\s*)(?:.*?))(\s*)(?:(?:(<!--)" r"((?:.*?\n?)*)(-->))|(\n|$))",
                bygroups(Text, Text, Keyword, Comment.MultiLine, Keyword, Text),
            ),
            (r"\s*\>\s*[^\s]*", Name.Constant),
            (
                r"(#+(?:\s*)(?:.*?))(\s*)(?:(?:(<!--)((?:.*?\n?)*)(-->))|(\n|$))",
                bygroups(
                    Generic.Heading, Text, Keyword, Comment.MultiLine, Keyword, Text
                ),
            ),
            (r"\s*\*\s*", Name.Variable.Magic, ("intent", "intent_rx")),
            (r".*\n", Text),
        ],
        "event": [include("comment"), (r"\s*(\n|$)", Text, "#pop")],
        "event_rx": [(r"({.*?})?", bygroups(using(JsonLexer)), "#pop")],
        "intent": [
            (r"\s*OR\s*", Keyword, "intent_rx"),
            include("comment"),
            (r"\s*(?:\n|$)", Text, "#pop"),
            default("#pop"),
        ],
        "intent_rx": [
            (r'["\'].*["\']', Name.Variable.Magic, "#pop"),
            (
                r"([^\s\{]*\s*)({.*?})?",
                bygroups(Name.Variable.Magic, using(JsonLexer)),
                "#pop",
            ),
            (r"\s*(\n|$)", Text, "#pop:2"),
        ],
    }
