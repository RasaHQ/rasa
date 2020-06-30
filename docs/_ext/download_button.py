from __future__ import absolute_import
from docutils import nodes
import jinja2
from docutils.parsers.rst.directives import unchanged
from docutils.parsers.rst import Directive


BUTTON_TEMPLATE = jinja2.Template(
    u"""
    <button 
        disabled
        class="button download__button"
    >
        Download project
    </button>
"""
)

# placeholder node for document graph
class download_button_node(nodes.General, nodes.Element):
    pass


class DownloadButtonDirective(Directive):
    required_arguments = 0

    option_spec = { }

    # this will execute when your directive is encountered
    # it will insert a button_node into the document that will
    # get visited during the build phase
    def run(self):
        env = self.state.document.settings.env
        app = env.app

        # app.add_stylesheet("button.css")

        node = download_button_node()
        return [node]


# build phase visitor emits HTML to append to output
def html_download_button_node(self, node):
    html = BUTTON_TEMPLATE.render()
    self.body.append(html)
    raise nodes.SkipNode


# if you want to be pedantic, define text, latex, manpage visitors too..


def setup(app):
    app.add_node(download_button_node, html=(html_download_button_node, None))
    app.add_directive("download-button", DownloadButtonDirective)
