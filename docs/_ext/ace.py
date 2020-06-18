from __future__ import absolute_import
from docutils import nodes
import jinja2
from docutils.parsers.rst.directives import unchanged
from docutils.parsers.rst import Directive

EDITOR_TEMPLATE = jinja2.Template(
    u"""
  <div id="ace-editor">
    {{code}}
  </div>
"""
)


# placeholder node for document graph
class editor_node(nodes.General, nodes.Element):
    pass


class CodeEditorDirective(Directive):
    has_content = True
    required_arguments = 0

    option_spec = {"code": unchanged}

    # this will execute when your directive is encountered
    # it will insert a copyable_node into the document that will
    # get visisted during the build phase
    def run(self):
        env = self.state.document.settings.env
        app = env.app

        # app.add_stylesheet('copyable.css')

        node = editor_node()
        node["code"] = u"\n".join(self.content)  # self.options['code']
        return [node]


# build phase visitor emits HTML to append to output
def html_visit_copyable_node(self, node):
    html = EDITOR_TEMPLATE.render(code=node["code"])
    self.body.append(html)
    raise nodes.SkipNode


# if you want to be pedantic, define text, latex, manpage visitors too..


def setup(app):
    app.add_node(editor_node, html=(html_visit_copyable_node, None))
    app.add_directive('code-editor', CodeEditorDirective)
