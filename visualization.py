from n0de import Node
import graphviz

def visualize_tree(tree: Node, target_name: str, label_names):
    leaf_names = label_names[target_name]

    def format_interval(interval):
        return f"[{interval[0]:.1f}, {interval[1]:.1f}]"

    def get_label(parent_feature, edge_label):
        if isinstance(label_names[parent_feature], list):
            # Handle intervals
            intervals = label_names[parent_feature][1]
            edge_label_int = int(edge_label.split()[-1])
            interval = intervals[edge_label_int]
            return format_interval(interval)
        else:
            # Handle categorical values
            value_dict = label_names[parent_feature]
            edge_label_int = int(edge_label.split()[-1])
            label = [key for key, value in value_dict.items() if value == edge_label_int]
            return f"{label[0]}" if label else edge_label

    def traverse(node: Node, graph, parent_name=None, edge_label="", parent=None):
        node_id = str(id(node))
        if node.is_leaf():
            node_label = f"Class: {[chave for chave, valor in leaf_names.items() if valor == node.leaf_value][0]}"
            node_color = 'lightblue'
        else:
            node_label = f"{node.feature}\nIG: {node.info_gain:.1f}"
            node_color = 'red'

        graph.node(node_id, label=node_label, style='filled', fillcolor=node_color, shape='box')

        if parent_name:
            label = get_label(parent.feature, edge_label)
            graph.edge(parent_name, node_id, f"= {label}", color='black')

        for child in node.children:
            traverse(child, graph, node_id, f"= {child.condition}", node)

    dot = graphviz.Digraph()
    traverse(tree, dot)
    return dot