"""
Helpers for dot graph file generation
"""
import typing


class DotGraph:

    def dot_subgraphs(self):
        return []

    def dot_repr(self):
        out = ""
        i = 0
        subgraphs = []
        for j, subgraph in enumerate(self.dot_subgraphs()):
            i, subgraph_code = subgraph.dot_repr(j, i)
            subgraphs.append(subgraph_code)
        graph = """digraph G {{
            {0:s}
        }}""".format("\n".join(subgraphs))
        return graph


class DotSubgraph(DotGraph):

    def dot_roots(self):
        return []

    def dot_label(self):
        return ""

    def dot_repr(self, j=0, i=0):
        defs = []
        links = []
        for root in self.dot_roots():
            i, these_defs, these_links = root.dot_repr(i)
            defs.extend(these_defs)
            links.extend(these_links)
        subgraph_label = self.dot_label()
        subgraph_code = """subgraph cluster_{0:d} {{
            {1:s}
            {2:s}
            {3:s}
        }}""".format(j, "\n".join(defs), "\n".join(links),
                     "label=\"{}\"".format(subgraph_label) if subgraph_label else "")
        return i, subgraph_code


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
                    links.append('{0:s} -> {1:s} [{2:s}];'
                                 .format(identifier, child_identifier, edge_set.get_prop_str()))
        return i, defs, links


class DotEdgeSet:

    def __init__(self, children, **props):
        self.children = children
        self.props = props

    def get_prop_str(self):
        return ", ".join('{}="{}"'.format(k, v) for k, v in self.props.items())
