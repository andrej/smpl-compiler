"""
Helpers for dot graph file generation
"""
import typing


class DotGraph:

    def dot_roots(self):
        return []

    def dot_repr(self):
        out = ""
        i = 0
        defs = []
        links = []
        for root in self.dot_roots():
            i, these_defs, these_links = root.dot_repr(i)
            defs.extend(these_defs)
            links.extend(these_links)
        graph = """digraph G {{
            {0:s}
            {1:s}
        }}
        """.format("\n".join(defs), "\n".join(links))
        return graph


class DotNode:

    def dot_edge_sets(self):
        return []

    def dot_label(self):
        return ""

    def dot_repr(self, i=0):
        to_discover = {self}
        to_draw = {}  # dict node -> graph_id

        while to_discover:
            node = to_discover.pop()
            identifier = "A_{}".format(i)
            to_draw[node] = identifier
            i += 1
            for edge_set in node.dot_edge_sets():
                for child in edge_set.children:
                    if child in to_draw:
                        continue  # avoid infinite loops on cycles
                    to_discover.add(child)

        defs = []
        links = []
        for node, identifier in to_draw.items():
            defs.append('{0:s} [shape=record, label="{1:s}"];'
                        .format(identifier, node.dot_label()))
            edge_sets = node.dot_edge_sets()
            for edge_set in edge_sets:
                for child in edge_set.children:
                    child_identifier = to_draw[child]
                    links.append('{0:s} -> {1:s} [label="{2:s}", color={3:s}];'
                                 .format(identifier, child_identifier, edge_set.label, edge_set.color))
        return i, defs, links


class DotEdgeSet:

    def __init__(self, children, label, color):
        self.children = children
        self.label = label
        self.color = color
