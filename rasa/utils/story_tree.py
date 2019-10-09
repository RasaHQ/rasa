"""This script generates a tree diagram from a story file."""

import argparse
import os
from termcolor import colored as term_colored
import pydoc
from math import log2
import re
import ast

#################################################
# Command line argument specification
#################################################

arg_parser = argparse.ArgumentParser(description="Represent story tree.")
arg_parser.add_argument("input", type=str, nargs="+", help="Input story file name(s)")
arg_parser.add_argument("--ambiguities", "-a", default="false", choices=["true", "false", "wizard"],
                        const="true", nargs="?", help="Display ambiguous branches only")
arg_parser.add_argument("--max-depth", "-m", type=int, default=None, help="Maximum depth")
arg_parser.add_argument("--coloring", "-c", default="role", choices=["role", "r", "ambiguities", "a", "depth", "none",
                                                                     "n"])
arg_parser.add_argument("--color-code", default="terminal", choices=["terminal", "markdown"])
arg_parser.add_argument("--branch", "-b", default="",
                        help="Restrict output to the given branch of the tree, separated by '/'")
arg_parser.add_argument("--stories", "-s", nargs="+", default=None,
                        help="Restrict output to the given stories.")
arg_parser.add_argument("--labels", "-l", action="store_true", default=False,
                        help="Show the names of stories at each node that has no siblings")
arg_parser.add_argument("--page", "-p", action="store_true", default=False,
                        help="Use pagination for output if necessary")
arg_parser.add_argument("--prune", default="most-visited", choices=["first", "last", "most-visited"],
                        const="First", nargs="?", help="Selection criterion for kept branches during pruning")
arg_parser.add_argument("--output", "-o", default="tree", choices=["tree", "stats", "pruned", "ok"],
                        nargs="+", help="What to display")
arg_parser.add_argument("--merge", nargs=1, help="Merge given story file into main file, avoiding ambiguities")

#################################################
# Helper functions
#################################################

color_code = "terminal"


def colored(string, color):
    # noinspection PyUnresolvedReferences
    global color_code
    if color_code == "terminal":
        return term_colored(string, color)
    elif color_code == "markdown":
        return f"<span style=\"color:{color}\">" + string + "</span>"
    elif color_code == "none":
        return string
    else:
        raise ValueError("Invalid color_code. Must be one of \"terminal\", \"markdown\", or \"none\"")


def slot_to_dict(string):
    regex_slot = r"- slot\{([^\}]*)\}$"  # Groups: slots
    regex_intent = r"\* ([^_\{]+)(?:\{([^\}]*)})?$"  # Groups: name | slots

    match = re.search(regex_slot, string)
    if match:
        return ast.literal_eval("{" + match.group(1) + "}")
    else:
        match = re.search(regex_intent, string)
        if match:
            if match.group(2):
                return ast.literal_eval("{" + match.group(2) + "}")
    return {}


#################################################
# Classes
#################################################

# ===============================================
# Node
# Represents a node in a tree graph
# ===============================================

class Node:

    def __init__(self, name="root", parent=None, story=""):
        self.count = 1
        self.name = name
        self.parent = parent
        self.children = []
        self.labels = [story]
        self._is_pruned = False

    def add_child(self, child):
        """
        Add the child node `child`, unless it exists already, in which case an error is raised.
        :type child: Node
        :param child: Child node
        """
        if self.get_child(child.name) is None:
            self.children.append(child)
        else:
            raise ValueError(f"A child with the name {child.name} already exists!")
        pass

    def get_child(self, name: str):
        """
        Get the child with the given name
        :param name: Name of the sought child
        :return: Child node
        """
        for child in self.children:
            if child.name == name:
                return child
        return None

    def print_string(self, branch="", stories=None, max_depth=None, only_ambiguous=False, show_labels=False,
                     coloring="role", include_users=True, _depth=0, _has_siblings=False):
        """
        Recursively generate a string representation of the tree with this node as root.
        :param stories: Restrict output to given stories
        :param branch: Restrict output to given branch (overwrites `stories`)
        :param max_depth: Go no deeper than this
        :param show_labels: Indicate branch labels on non-branch points
        :param only_ambiguous: Only output ambiguous branches
        :param coloring: Coloring rule ('role', 'depth', 'ambiguities')
        :param _depth: Recursion depth - only set in recursion step!
        :param include_users: When `only_ambiguous` is `True`, include ambiguous user responses
        :param _has_siblings: True, iff this node has siblings - only set in recursion step!
        :return: The generated string
        """

        # Abort recursion if max depth is reached
        if max_depth and _depth >= max_depth:
            return ""

        # Decide how to color the present node
        if coloring[0] == "r":  # "role"
            color = {"S": "yellow", "U": "blue", "W": "green"}.get(self.name[0])
        elif coloring[0] == "d":  # "depth"
            color = {1: "green", 2: "magenta", 3: "yellow", 4: "cyan", 5: "blue", 6: "grey"}.get(_depth, "grey")
        elif coloring[0] == "n":  # "none"
            color = "none"
        elif coloring[0] == "a":  # "ambiguities"
            if _has_siblings:
                color = "red"
            elif self.has_descendants_with_siblings:
                color = "yellow"
            else:
                color = "white"
        else:
            raise ValueError(f"Invalid coloring \"{coloring}\". Must be one of 'roles', 'depth', "
                             f"'ambiguities' or 'none'.")

        # If only ambiguous nodes should be printed, then print only if there are siblings or descendants with siblings
        if (not only_ambiguous) or _has_siblings or self.has_descendants_with_siblings(include_users):

            # Visit count indicator for non-root nodes only
            count_str = f" ({self.count})" if _depth > 0 else ""

            # Show branch labels iff visit count is 1
            if show_labels and self.count == 1:
                result = "+" + "-" * (2 * _depth) + " " + colored(self.name + count_str, color) \
                         + f" <{self.labels[0]}>" + os.linesep
            else:
                result = "+" + "-" * (2 * _depth) + " " + colored(self.name + count_str, color) \
                         + os.linesep
        else:
            # We show only ambiguous branches, and this node is not root of an ambiguous branch
            result = ""

        # Prepare _has_siblings for recursion step
        has_siblings = (len(self.children) > 1)
        if has_siblings and not include_users:
            all_children_are_users = all(child.name.startswith("U:") or child.name.startswith("S:") for child in self.children)
            has_siblings = not all_children_are_users

        # Recursion step into all child nodes
        if branch:
            # Output should be restricted to `branch`
            path = branch.split("/")            # Split the branch spec into one for each level
            sought_name = path[0]               # First entry is where we should step into now
            remain_branch = "/".join(path[1:])  # Remaining entries have to be passed on to recursive call

            for child in self.children:
                if child.name == sought_name or sought_name == "*":
                    result += child.print_string(branch=remain_branch,
                                                 stories=stories,
                                                 max_depth=max_depth,
                                                 only_ambiguous=only_ambiguous,
                                                 show_labels=show_labels,
                                                 include_users=include_users,
                                                 coloring=coloring,
                                                 _depth=_depth + 1,
                                                 _has_siblings=has_siblings)
        else:
            # No branch restriction -> step into all child branches unless stories are restricted
            for child in self.children:
                if stories is None or not set(child.labels).isdisjoint(stories):
                    result += child.print_string(branch=branch,
                                                 stories=stories,
                                                 max_depth=max_depth,
                                                 only_ambiguous=only_ambiguous,
                                                 show_labels=show_labels,
                                                 include_users=include_users,
                                                 coloring=coloring,
                                                 _depth=_depth + 1,
                                                 _has_siblings=has_siblings)
        return result

    def has_descendants_with_siblings(self, include_users):
        """
        Boolean that indicates if there are any descendants that have siblings.
        :return: True, iff a descendant node has siblings
        """
        if len(self.children) > 1:
            if include_users:
                return True
            else:
                all_children_are_users = all(child.name.startswith("U:") or child.name.startswith("S:") for child in self.children)
                if all_children_are_users:
                    return any(child.has_descendants_with_siblings(include_users) for child in self.children)
                else:
                    return True
        elif len(self.children) == 1:
            return list(self.children)[0].has_descendants_with_siblings(include_users)
        else:
            return False

    def __str__(self):
        return self.print_string()

    def prune(self, keep: str):
        """Removes all ambiguous branches"""
        if self._is_pruned:
            return
        if len(self.children) > 0:
            if len(self.children) > 1:
                if any([child.name.startswith("W:") for child in self.children]):
                    if keep == "first":
                        del self.children[1:]
                    elif keep == "last":
                        del self.children[:-1]
                    elif keep == "most-visited":
                        visit_counts = [len(c.labels) for c in self.children]
                        keep_idx = visit_counts.index(max(visit_counts))
                        # Delete all but the one at `keep_idx`
                        del self.children[:keep_idx]
                        if len(self.children) > 1:
                            del self.children[1:]
                    else:
                        raise ValueError("Invalid prune keep criterion.")
            for child in self.children:
                child.prune(keep)
        self._is_pruned = True

    def remove(self, story) -> bool:
        """Remove the given story from this node and recursively from all
        descendants. """
        if story in self.labels:
            # Remove the story from internal stats
            self.labels = [label for label in self.labels if label != story]
            # Recurse through all children
            new_children = []
            for child in self.children:
                if not child.remove(story):
                    # Only retain children that did not self-delete
                    new_children.append(child)
                else:
                    # Delete this child node
                    self.count -= 1
                    del child

            self.children = new_children

        assert (len(self.labels) == 0) == (self.count == 0)

        # If this node had no other stories than the one we just
        # deleted, then return True, and False otherwise
        return len(self.labels) == 0

    @property
    def leafs(self):
        leafs = set()

        # noinspection PyUnusedLocal
        def callback_discover_leaf(node, *args):
            assert len(node.labels) >= 1, f"Leaf has no story assigned!"
            # Leafs may have multiple stories assigned, iff stories have duplicates
            # Ignore duplicates iff the tree was pruned
            if self._is_pruned:
                leafs.add(node.labels[0])
            else:
                for story in node.labels:
                    leafs.add(story)
            return {}

        self._depth_first_search({"discover_leaf": callback_discover_leaf}, {})

        return leafs

    @property
    def duplicates(self):
        duplicates = []

        # noinspection PyUnusedLocal
        def callback_discover_leaf(node, *args):
            assert len(node.labels) >= 1, f"Leaf has no story assigned!"
            # Leafs have multiple stories assigned, iff stories have duplicates
            if len(node.labels) > 1:
                duplicates.append(node.labels)
            return {}

        self._depth_first_search({"discover_leaf": callback_discover_leaf}, {})

        return duplicates

    def stats(self):
        """
        Collects statistics about the tree that has this node as a root.
        :return: Dict with statistical information
        """
        statistics = {
            "num_nodes": 0,  # Total number of nodes in the tree (steps in all dialogues)
            "num_nodes_with_multiple_children": 0,  # Number of nodes that have multiple children
            "num_leaves": 0,  # How many stories are present?
            "depth": 0,  # How deep is the graph
            "ambiguity_depth": 0,  # How deep is the deepest branch point?
            "ambiguity_chain_length": 0,  # How many branch points follow each other (max)?
            "ambiguity_level": 0,  # How many leaves are connected to root via branch points?
            "story_stats": {}  # Stats about individual stories
        }

        def callback_discover_node(node, depth, flags):
            statistics["num_nodes"] += 1
            if len(node.children) > 1:
                if any([child.name.startswith("W:") for child in node.children]):
                    statistics["num_nodes_with_multiple_children"] += 1
                    statistics["ambiguity_depth"] = max(statistics["ambiguity_depth"], depth)
                    statistics["ambiguity_chain_length"] = max(statistics["ambiguity_chain_length"],
                                                               flags["ambiguity_chain_length"] + 1)
                    if flags["linear_so_far"]:
                        statistics["ambiguity_level"] += node.count
                    for story in node.labels:
                        if story in statistics["story_stats"]:
                            statistics["story_stats"][story]["ambiguity_length"] += 1
                            statistics["story_stats"][story]["related_to"].update(node.labels)
                        else:
                            statistics["story_stats"][story] = {
                                "length": depth,
                                "ambiguity_length": 1,
                                "related_to": set(node.labels)
                            }
                    return {
                        "linear_so_far": False,
                        "ambiguity_chain_length": flags["ambiguity_chain_length"] + 1
                    }
            return {}

        # noinspection PyUnusedLocal
        def callback_discover_leaf(node, depth, flags):
            statistics["num_leaves"] += 1
            statistics["depth"] = max(statistics["depth"], depth)
            story = node.labels[0]
            if story in statistics["story_stats"]:
                statistics["story_stats"][story]["length"] = depth
            return {}

        self._depth_first_search({
            "discover_node": callback_discover_node,
            "discover_leaf": callback_discover_leaf
        }, flags={"linear_so_far": True, "ambiguity_chain_length": 0})

        return statistics

    def _depth_first_search(self, callbacks, flags, _depth=0):
        new_flags = flags.copy()
        if len(self.children) == 0 and "discover_leaf" in callbacks:
            new_flags.update(callbacks["discover_leaf"](self, _depth, flags))
            return
        for child in self.children:
            if "discover_node" in callbacks:
                new_flags.update(callbacks["discover_node"](child, _depth + 1, flags))
            # noinspection PyProtectedMember
            child._depth_first_search(callbacks, new_flags, _depth + 1)


# ===============================================
# Tree
# Represents a tree graph
# ===============================================

class Tree:

    def __init__(self):
        self.root = Node()         # Root node, should never change
        self.pointer = self.root   # Pointer to the currently active node
        self.label = ""            # Label for active branch

    def add_or_goto(self, name):
        """
        If a branch with name `name` is a child of the currently active node, then move `self.pointer`
        to that branch and update visit counts and branch name lists. Otherwise, create a new child
        branch with this name and move the pointer to it.
        :param name: Name of the (new) branch to go to
        :return: True, iff a new branch was created
        """
        # Check if branch with name `name` exists
        for branch in self.pointer.children:
            if branch.name == name:
                branch.count += 1              # Increase visit count
                branch.labels += [self.label]  # Append new branch label
                self.pointer = branch          # Move pointer to this branch
                return False

        # Add a new branch
        new_branch = Node(name, parent=self.pointer, story=self.label)
        self.pointer.add_child(new_branch)
        self.pointer = new_branch
        return True

    def adding_creates_ambiguity(self, name: str):
        """
        Returns True iff adding a branch with this name would result in an ambiguity in this tree,
        i.e. another child node exists, which is as Wizard node.
        :param name: Name of the branch (user/wizard action)
        :return: True iff ambiguous
        """
        return name.startswith("W") and any(c.name.startswith("W") for c in self.pointer.children)

    def up(self):
        """
        Move the active branch pointer one step towards root
        :return: True, iff active branch is not already on root
        """
        if self.pointer != self.root:
            self.pointer = self.pointer.parent
            return True
        else:
            return False

    def reset(self, story):
        """
        Reset the active branch pointer to root and specify a new story label to use in `self.add_or_goto`.
        :param story: New story label
        """
        self.pointer = self.root
        self.label = story

    def remove(self, story=None):
        """
        Remove the given story, or the story with the name stored in self.label
        :param story: Name of the story
        """
        if story:
            self.root.remove(story)
        else:
            self.root.remove(self.label)

    def to_string(self, branch="", max_depth=None, show_labels=False, only_ambiguous=False, coloring="role",
                  include_users=True, stories=None):
        """
        Create a string representation of the tree.
        :param stories: Restrict output to given stories
        :param branch: Restrict output to given branch (overwrites `stories`)
        :param max_depth: Go no deeper than this
        :param show_labels: Indicate branch labels on non-branch points
        :param only_ambiguous: Only output ambiguous branches
        :param coloring: Coloring rule ('role', 'depth', 'ambiguities')
        :param include_users: When `only_ambiguous` is `True`, include ambiguous user responses
        :return: The generated string
        """
        return self.root.print_string(branch=branch, max_depth=max_depth, include_users=include_users,
                                      show_labels=show_labels, only_ambiguous=only_ambiguous, coloring=coloring,
                                      stories=stories)

    def __str__(self):
        return self.root.print_string()

    def prune(self, *args):
        self.root.prune(*args)

    @property
    def leafs(self):
        return self.root.leafs

    @property
    def duplicates(self):
        return self.root.duplicates

    def stats(self) -> dict:
        """
        Compute statistics about this tree.
        :return: The generated dict with statistical information
        """
        return self.root.stats()


#################################################
# Main
#################################################

if __name__ == '__main__':

    def main():

        stats = None

        # Read command line arguments
        args = arg_parser.parse_args()
        story_file_names = args.input  # Input file name
        global color_code
        color_code = args.color_code  # "terminal" / "markdown"

        # Generate the story tree
        n = 0
        tree = Tree()
        slots = {}
        for story_file_name in story_file_names:
            with open(story_file_name, "r") as story_file:
                for line in story_file:
                    if line.startswith("##"):
                        n += 1
                        tree.reset(story=line[2:].strip())
                        slots.clear()
                    else:
                        if line.lstrip().startswith("*"):
                            name = "U: "
                        elif line.lstrip().startswith("- slot"):
                            name = "S: "
                        else:
                            name = "W: "

                        if name in ["U: ", "S: "]:
                            # Slots might have been updated -> keep track of it
                            new_slots = slot_to_dict(line)
                            copy_slots = slots.copy()
                            copy_slots.update(new_slots)
                            if copy_slots.items() == slots.items():
                                # Setting this slot does not change anything
                                if name == "S: ":
                                    # Ignore redundant slot lines
                                    name = ""
                                    line = ""
                            else:
                                slots.update(new_slots)

                        name += line.strip()[2:]
                        if line.strip():
                            tree.add_or_goto(name)

        # Merge other story file (only take in stories that don't create ambiguities)
        if args.merge:
            successful_merge = []  # Stories that got merged in successfully
            total_num_merge = 0    # Total number of stories that should have been merged
            for story_file_name in args.merge:
                with open(story_file_name, "r") as story_file:
                    active_story = ""
                    for line in story_file:
                        if line.startswith("##"):
                            if active_story:
                                # The previous story was merged all the way and
                                # thus `active_story` was not set to `""`. In this
                                # case, we remember the name of the story that
                                # merged successfully
                                successful_merge.append(active_story)
                            total_num_merge += 1
                            n += 1
                            active_story = line[2:].strip()
                            tree.reset(story=active_story)
                        else:
                            if active_story:
                                if line.lstrip().startswith("*"):
                                    name = "U: "
                                elif line.lstrip().startswith("- slot"):
                                    name = "S: "
                                else:
                                    name = "W: "
                                name += line.strip()[2:]
                                if line.strip():
                                    if tree.adding_creates_ambiguity(name):
                                        # Merging `active_story` would create ambiguity
                                        tree.remove()
                                        active_story = ""
                                    else:
                                        tree.add_or_goto(name)

        # Display the tree if required
        if "tree" in args.output:
            _print = pydoc.pager if args.page else print
            _print(tree.to_string(only_ambiguous=(args.ambiguities in ["true", "wizard"]),
                                  include_users=(args.ambiguities != "wizard"),
                                  max_depth=args.max_depth,
                                  show_labels=args.labels,
                                  coloring=args.coloring,
                                  branch=args.branch,
                                  stories=args.stories))

        # Display statistics if required
        if "stats" in args.output:
            stats = tree.stats()
            duplicates = tree.duplicates

            print()
            print(colored("Text summary:", "cyan"))
            if duplicates:
                print(f"The input contains {stats['num_leaves']} stories, but there are some duplicates (see below).")
            else:
                print(f"The input contains {stats['num_leaves']} unique stories.")
            print(f"The longest story is {stats['depth']} nodes deep.")
            print(
                f"{stats['num_nodes_with_multiple_children']} / {stats['num_nodes']} = "
                f"{100.0 * stats['num_nodes_with_multiple_children'] / stats['num_nodes']:.2f}% of all nodes "
                f"have multiple children.")
            print(f"The deepest branch point occurs after {stats['ambiguity_depth']} steps.")
            print(f"We encounter up to {stats['ambiguity_chain_length']} branch points in a single story.")
            print(f"{stats['ambiguity_level']} / {stats['num_leaves']} = "
                  f"{100.0 * stats['ambiguity_level'] / stats['num_leaves']:.2f}% of all stories are ambiguous.")
            if args.merge:
                # noinspection PyUnboundLocalVariable
                print(f"Successfully merged {len(successful_merge)} out of {total_num_merge} stories.")

            if duplicates:
                print()
                print(colored("Duplicate stories:", "cyan"))
                for d in duplicates:
                    print(d)

            print()
            print(colored("Statistics table:", "cyan"))
            print(f"num stories:      {stats['num_leaves']}")
            print(f"max turns:        {stats['depth']}")
            print(f"num nodes:        {stats['num_nodes']}")
            print(f"branch-points:    {stats['num_nodes_with_multiple_children']} "
                  f"({100.0 * stats['num_nodes_with_multiple_children'] / stats['num_nodes']:.2f}%)")
            print(f"ambiguity depth:  {stats['ambiguity_depth']}")
            print(f"ambiguity length: {stats['ambiguity_chain_length']}")
            print(f"ambiguity level:  {stats['ambiguity_level']} "
                  f"({100.0 * stats['ambiguity_level'] / stats['num_leaves']:.2f}%)")
            if stats['ambiguity_level'] > 0.0:
                print(f"ambiguity log:    log2({stats['ambiguity_level']}) = {log2(stats['ambiguity_level']):.2f} ")

            tree.prune(args.prune)
            pruned_stats = tree.stats()
            print()
            print(colored("After pruning:", "cyan"))
            print(f"num stories:      {pruned_stats['num_leaves']} "
                  f"({stats['num_leaves'] - pruned_stats['num_leaves']} fewer)")
            print(f"max turns:        {pruned_stats['depth']} "
                  f"({stats['depth'] - pruned_stats['depth']} fewer)")
            print(f"num nodes:        {pruned_stats['num_nodes']} "
                  f"({stats['num_nodes'] - pruned_stats['num_nodes']} fewer)")

            if len(stats["story_stats"]) > 0:
                print()
                print(colored("Most ambiguous stories:", "cyan"))
                print(f"{'Story':>15}", f"{'# relations':>14}", f"{'# branchings':>14}", f"{'# turns':>14}")
                for story, values in sorted(stats["story_stats"].items(),
                                            key=lambda kv: [
                                                len(kv[1]["related_to"]),
                                                kv[1]["ambiguity_length"],
                                                kv[1]["length"]],
                                            reverse=True)[:12]:
                    print(f"{story:>15} {len(values['related_to']):>14} {values['ambiguity_length']:>14} "
                          f"{values['length']:>14}")

            print()

        # Display remaining stories after pruning, if required
        if "pruned" in args.output:
            if args.merge:
                for story in sorted(successful_merge):
                    print(story)
            else:
                tree.prune(args.prune)
                for story in sorted(tree.leafs):
                    print(story)

        if "ok" in args.output:
            if not stats:
                stats = tree.stats()
            if stats['num_nodes_with_multiple_children'] > 0:
                print("False")
            else:
                print("True")

    main()
