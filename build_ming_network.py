import csv
import json
from pathlib import Path

import qhchina
qhchina.load_fonts()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def load_graph(node_path, edge_path):
    """Load node and edge CSV files into a NetworkX graph."""
    graph = nx.Graph()

    with open(node_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            graph.add_node(
                row["id"],
                label=row.get("label", row["id"]),
                degree=int(row.get("degree", 0)),
            )

    with open(edge_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source = row["source"]
            target = row["target"]
            weight = int(row.get("weight", 1))
            if source == target:
                continue
            graph.add_edge(source, target, weight=weight)

    return graph


def compute_louvain_clusters(graph):
    """Return community memberships using NetworkX's Louvain implementation."""
    if not hasattr(nx.algorithms.community, "louvain_communities"):
        raise RuntimeError("This NetworkX version does not include Louvain communities.")

    communities = nx.algorithms.community.louvain_communities(
        graph,
        weight="weight",
        seed=42,
    )
    return communities


def cluster_color_map(graph, communities):
    """Map each node to a color based on its Louvain community."""
    node_to_cluster = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_cluster[node] = idx

    cluster_ids = sorted(set(node_to_cluster.values()))
    palette = {cluster_id: plt.cm.tab10(index % 10) for index, cluster_id in enumerate(cluster_ids)}
    node_colors = [palette[node_to_cluster[node]] for node in graph.nodes()]
    return node_to_cluster, node_colors


def render_network(graph, output_path, title, layout="spring"):
    """Render the graph and save it to disk."""
    plt.rcParams["axes.unicode_minus"] = False

    if layout == "spring":
        pos = nx.spring_layout(graph, seed=42, k=1.8)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    fig, ax = plt.subplots(figsize=(12, 8))
    communities = compute_louvain_clusters(graph)
    node_to_cluster, node_colors = cluster_color_map(graph, communities)

    # Eigenvector centrality determines node size. Use the standard implementation
    # because the expanded roster may produce a disconnected graph.
    centrality = nx.eigenvector_centrality(graph, weight="weight", max_iter=1000)
    node_sizes = [max(500.0, 2200.0 * centrality[node]) for node in graph.nodes()]

    nx.draw_networkx_edges(
        graph,
        pos,
        width=[0.6 + 0.5 * d["weight"] for _, _, d in graph.edges(data=True)],
        alpha=0.7,
        edge_color="gray",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.8,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, font_size=16, ax=ax)

    ax.set_axis_off()
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return node_to_cluster, centrality


def build_cooccurrence_graph(text_path, generals):
    """Create a co-occurrence graph from a text file where people appearing on the same line are connected."""
    graph = nx.Graph()
    for person in generals:
        graph.add_node(person, label=person, degree=0)

    with open(text_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    for line in lines:
        present = [person for person in generals if person in line]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] = graph[a][b].get("weight", 1) + 1
                else:
                    graph.add_edge(a, b, weight=1)

    for node, degree in graph.degree():
        graph.nodes[node]["degree"] = degree

    isolated = [node for node, degree in graph.degree() if degree == 0]
    if isolated:
        graph.remove_nodes_from(isolated)

    for node, degree in graph.degree():
        graph.nodes[node]["degree"] = degree

    return graph


def write_graph_csvs(graph, node_path, edge_path):
    """Export node and edge tables as CSV files."""
    with open(node_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "degree"])
        writer.writeheader()
        for node in sorted(graph.nodes()):
            writer.writerow({
                "id": node,
                "label": graph.nodes[node].get("label", node),
                "degree": graph.degree(node),
            })

    with open(edge_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "weight"])
        writer.writeheader()
        for source, target, data in sorted(graph.edges(data=True), key=lambda x: (x[0], x[1])):
            writer.writerow({
                "source": source,
                "target": target,
                "weight": data.get("weight", 1),
            })


def write_network_html(graph, node_to_cluster, output_path):
    """Write a self-contained interactive network visualization."""
    layout = nx.spring_layout(graph, seed=42, k=1.8, weight="weight")
    min_x = min(point[0] for point in layout.values())
    max_x = max(point[0] for point in layout.values())
    min_y = min(point[1] for point in layout.values())
    max_y = max(point[1] for point in layout.values())
    scale = min(800 / (max_x - min_x), 540 / (max_y - min_y))
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    positions = {
        node: {
            "x": round(500 + (point[0] - center_x) * scale, 2),
            "y": round(350 - (point[1] - center_y) * scale, 2),
        }
        for node, point in layout.items()
    }
    palette = [
        "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
        "#ec4899", "#14b8a6", "#84cc16", "#f97316", "#6366f1",
    ]
    cluster_ids = sorted(set(node_to_cluster.values()))
    cluster_colors = {
        cluster_id: palette[index % len(palette)]
        for index, cluster_id in enumerate(cluster_ids)
    }
    nodes = []
    for node in sorted(graph.nodes()):
        degree = graph.degree(node)
        nodes.append({
            "id": node,
            "label": graph.nodes[node].get("label", node),
            "degree": degree,
            **positions[node],
            "color": cluster_colors[node_to_cluster[node]],
            "explanation": (
                f"{node} is a Ming dynasty general in the co-occurrence network. "
                f"Degree centrality is {degree}. This node represents how frequently "
                "the general appears with others in the same line of the source text."
            ),
            "attributes": [
                ["Name", node],
                ["Degree", str(degree)],
                ["Node type", "General"],
            ],
        })

    edges = []
    for source, target, data in sorted(graph.edges(data=True)):
        weight = data.get("weight", 1)
        explanation = (
            f"{source} and {target} appear together {weight} times in the same text "
            "line, which indicates a repeated association in the Ming narrative."
        )
        edges.append({
            "source": source,
            "target": target,
            "weight": weight,
            "explanation": explanation,
            "attributes": [
                ["Source", source],
                ["Target", target],
                ["Weight", str(weight)],
                ["Interpretation", explanation],
            ],
        })

    node_json = json.dumps(nodes, ensure_ascii=False).replace("<", "\\u003c")
    edge_json = json.dumps(edges, ensure_ascii=False).replace("<", "\\u003c")
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ming Dynasty Generals Network</title>
  <style>
    :root { color-scheme: light; --bg: #f3f6fb; --panel: #fff; --border: #dfe7f1; --text: #1f2937; --muted: #5b6472; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Segoe UI", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; background: var(--bg); color: var(--text); }
    .layout { display: grid; grid-template-columns: minmax(0, 1fr) 340px; min-height: 100vh; }
    #network { min-height: 600px; background: linear-gradient(180deg, #fff 0%, #eef4fb 100%); border-right: 1px solid var(--border); }
    svg { display: block; width: 100%; height: 100%; min-height: 600px; }
    .panel { padding: 20px 18px; background: var(--panel); }
    h1 { margin: 0 0 14px; font-size: 1.2rem; }
    .box { margin-bottom: 16px; padding: 14px 16px; border: 1px solid var(--border); border-radius: 12px; background: #f8fafc; }
    .section-label { margin-bottom: 8px; color: var(--muted); font-size: .7rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .details { color: var(--text); font-size: .95rem; line-height: 1.6; overflow-wrap: anywhere; }
    .row { margin-bottom: 9px; line-height: 1.5; }
    .summary { color: var(--muted); font-size: .9rem; line-height: 1.5; }
    button { margin: 0 0 16px; padding: 10px 14px; border: 0; border-radius: 8px; background: #2563eb; color: white; font: inherit; cursor: pointer; }
    button:hover { background: #1d4ed8; }
    .edge-line { stroke: #9aa0a6; stroke-width: 1.5; opacity: .75; cursor: pointer; }
    .edge-label { fill: #374151; font-size: 11px; text-anchor: middle; dominant-baseline: middle; }
    .node-circle { stroke: #1d4ed8; stroke-width: 1.2; cursor: pointer; }
    .node-label { fill: #111827; font-size: 14px; text-anchor: middle; dominant-baseline: middle; pointer-events: none; }
    @media (max-width: 800px) {
      .layout { grid-template-columns: 1fr; }
      #network { min-height: 65vh; border-right: 0; border-bottom: 1px solid var(--border); }
      svg { min-height: 65vh; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <main id="network" aria-label="Interactive network graph"></main>
    <aside class="panel">
      <h1>Ming Dynasty Generals Network</h1>
      <button id="download-html" type="button">Download this HTML</button>
      <div class="box">
        <div class="section-label">Selected item</div>
        <div id="details" class="details">Click a node or edge to inspect its details.</div>
      </div>
      <div class="box">
        <div class="section-label">Legend</div>
        <div class="summary">Node colors identify Louvain communities. Larger nodes indicate more connections; thicker edges indicate stronger ties.</div>
      </div>
    </aside>
  </div>
  <script>
    const nodeData = __NODE_DATA__;
    const edgeData = __EDGE_DATA__;
    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 1000 700");
    document.getElementById("network").appendChild(svg);

    const positions = {};
    nodeData.forEach(node => {
      positions[node.id] = { x: node.x, y: node.y };
    });

    const edgeLayer = document.createElementNS(svgNS, "g");
    const nodeLayer = document.createElementNS(svgNS, "g");
    svg.append(edgeLayer, nodeLayer);
    edgeData.forEach(edge => {
      const source = positions[edge.source], target = positions[edge.target];
      if (!source || !target) return;
      const line = document.createElementNS(svgNS, "line");
      line.setAttribute("x1", source.x); line.setAttribute("y1", source.y);
      line.setAttribute("x2", target.x); line.setAttribute("y2", target.y);
      line.setAttribute("class", "edge-line");
      line.setAttribute("stroke-width", String(1.2 + edge.weight * 0.25));
      line.addEventListener("click", () => showDetails("Relationship", edge));
      edgeLayer.appendChild(line);
    });
    nodeData.forEach(node => {
      const pos = positions[node.id];
      const circle = document.createElementNS(svgNS, "circle");
      circle.setAttribute("cx", pos.x); circle.setAttribute("cy", pos.y);
      circle.setAttribute("r", String(12 + node.degree * 0.7));
      circle.setAttribute("fill", node.color); circle.setAttribute("class", "node-circle");
      circle.addEventListener("click", () => showDetails("General", node));
      nodeLayer.appendChild(circle);
      const label = document.createElementNS(svgNS, "text");
      label.setAttribute("x", pos.x); label.setAttribute("y", pos.y + 2);
      label.setAttribute("class", "node-label"); label.textContent = node.label;
      nodeLayer.appendChild(label);
    });

    function showDetails(type, item) {
      const details = document.getElementById("details");
      details.replaceChildren();
      const heading = document.createElement("div");
      heading.className = "section-label"; heading.textContent = type;
      const explanation = document.createElement("div");
      explanation.textContent = item.explanation;
      details.append(heading, explanation);
      item.attributes.forEach(([key, value]) => {
        const row = document.createElement("div");
        row.className = "row";
        const label = document.createElement("strong");
        label.textContent = key + ": ";
        row.append(label, document.createTextNode(value));
        details.appendChild(row);
      });
    }

    document.getElementById("download-html").addEventListener("click", () => {
      const content = "<!DOCTYPE html>\\n" + document.documentElement.outerHTML;
      const link = document.createElement("a");
      link.href = URL.createObjectURL(new Blob([content], { type: "text/html;charset=utf-8" }));
      link.download = "ming_generals_network.html";
      link.click();
      URL.revokeObjectURL(link.href);
    });
  </script>
</body>
</html>
"""
    html = html.replace("__NODE_DATA__", node_json).replace("__EDGE_DATA__", edge_json)
    Path(output_path).write_text(html, encoding="utf-8")


def main():
    base_dir = Path(".")
    node_path = base_dir / "ming_generals_nodes.csv"
    edge_path = base_dir / "ming_generals_edges.csv"
    image_path = base_dir / "ming_generals_louvain_forced.png"
    html_path = base_dir / "ming_generals_network.html"

    generals = [
        "徐達",
        "常遇春",
        "俞通海",
        "湯和",
        "鄧愈",
        "李文忠",
        "馮勝",
        "傅友德",
        "胡大海",
        "沐英",
        "藍玉",
        "廖永忠",
        "朱亮祖",
        "周德興",
        "郭英",
        "吳良",
        "王弼",
        "唐勝宗",
        "陳桓",
        "耿炳文",
        "郭子興",
        "張德勝",
        "朱棣",
        "李善長",
        "費聚",
        "康茂才",
        "何文輝",
        "楊璟",
        "王保保",
        "張輔",
    ]

    graph = build_cooccurrence_graph(base_dir / "mingshi.txt", generals)
    write_graph_csvs(graph, node_path, edge_path)
    node_to_cluster, centrality = render_network(
        graph,
        image_path,
        "Ming Dynasty Generals Network (Louvain, Force-Directed, Eigenvector)",
        layout="spring",
    )
    write_network_html(graph, node_to_cluster, html_path)

    print("Eigenvector centrality:")
    for node in sorted(centrality, key=lambda n: centrality[n], reverse=True):
        print(f"{node}: {centrality[node]:.4f}")
    print(f"Generated {node_path}, {edge_path}, {image_path}, and {html_path}")


if __name__ == "__main__":
    main()
