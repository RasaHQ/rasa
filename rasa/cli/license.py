import argparse
from typing import List

import rasa.shared.utils.cli
from rasa.cli import SubParsersAction


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add license parser.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    license_parser = subparsers.add_parser(
        "license",
        parents=parents,
        help="Displays licensing information.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Display licensing information.",
    )
    license_parser.set_defaults(func=display_license_information)


def display_license_information(_: argparse.Namespace) -> None:
    """Display licensing information to stdout."""
    rasa.shared.utils.cli.print_info(
        "By installing and using this software, you agree to be "
        "bound by the terms and conditions of the Developer Terms "
        "available at https://rasa.com/developer-terms. "
        "Please review the Developer Terms carefully before proceeding.\n\n"
        "Rasa Pro relies on several 3rd-party dependencies. "
        "The ones below require a license disclorure:\n",
        PSYCOPG2_LICENSE_DISCLOSURE,
    )


PSYCOPG2_LICENSE_DISCLOSURE = """
psycopg2 & psycopg2-binary
--------------------------

psycopg2 and the LGPL

psycopg2 is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

psycopg2 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

In addition, as a special exception, the copyright holders give permission to link this program with the OpenSSL library (or with modified versions of OpenSSL that use the same license as OpenSSL), and distribute linked combinations including the two.

You must obey the GNU Lesser General Public License in all respects for all of the code used other than OpenSSL. If you modify file(s) with this exception, you may extend this exception to your version of the file(s), but you are not obligated to do so. If you do not wish to do so, delete this exception statement from your version. If you delete this exception statement from all source files in the program, then also delete it here.

You should have received a copy of the GNU Lesser General Public License along with psycopg2 (see the doc/ directory.) If not, see https://www.gnu.org/licenses/. Alternative licenses

The following BSD-like license applies (at your option) to the files following the pattern psycopg/adapter*.{h,c} and psycopg/microprotocol*.{h,c}:

Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it and redistribute it freely, subject to the following restrictions:

The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.

Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.

This notice may not be removed or altered from any source distribution.
"""  # noqa: E501
