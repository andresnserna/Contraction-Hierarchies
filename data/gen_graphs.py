import argparse, os, random


def erdos_renyi(n, p, directed, seed=None):
    rng = random.Random(seed)
    nodes = {i for i in range(n)}
    edges = {}

    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:

                weight = rng.randint(1, 20)
                if directed:
                    edges[(u, v)] = {
                        "weight": weight,
                    }
                else:
                    edges[(u, v)] = {
                        "weight": weight
                    }
                    edges[(v, u)] = {
                        "weight": weight
                    }

    return nodes, edges


def cycle(n, directed, seed=None):
    rng = random.Random(seed)
    nodes = {i for i in range(n)}
    edges = {}

    for u in range(n):
        v = (u + 1) % n
        if directed:
            edges[(u, v)] = {
                "weight": rng.randint(1, 20),
            }
        else:
            weight = rng.randint(1, 20)
            edges[(u, v)] = {
                "weight": weight
            }
            edges[(v,u)] = {
                "weight": weight
            }

    return nodes, edges


def complete_graph(n, directed, seed=None):
    rng = random.Random(seed)
    nodes = {i for i in range(n)}
    edges = {}

    for u in range(n):
        for v in range(n):
            if u != v:
                weight = rng.randint(1, 20)
                edges[(u, v)] = {
                    "weight": weight
                }
                if not directed:
                    edges[(v, u)] = {
                        "weight": weight
                    }

    return nodes, edges


def path_graph(n, directed, seed=None):
    rng = random.Random(seed)
    nodes = {i for i in range(n)}
    edges = {}

    for u in range(n - 1):
        v = u + 1
        weight = rng.randint(1, 20)
        edges[(u, v)] = {
            "weight": weight
        }
        if not directed:
            edges[(v, u)] = {
                "weight": weight
            }

    return nodes, edges


def random_trees(n, directed, seed=None):
    rng = random.Random(seed)
    nodes = {i for i in range(n)}
    edges = {}

    for u in range(1, n):
        v = rng.randint(0, u - 1)
        weight = rng.randint(1, 20)
        edges[(u, v)] = {
            "weight": weight
        }
        if not directed:
            edges[(v, u)] = {
                "weight": weight
            }

    return nodes, edges


def balanced_trees(n, degree, directed, seed=None):
    rng = random.Random(seed)
    nodes = {i for i in range(n)}
    edges = {}

    for u in range(n):
        for i in range(1, degree + 1):
            v = degree * u + i
            if v < n:
                weight = rng.randint(1, 20)
                edges[(u, v)] = {
                    "weight": weight
                }
                if not directed:
                    edges[(v, u)] = {
                        "weight": weight
                    }

    return nodes, edges


def grid_graph(n, directed, seed=None):
    rng = random.Random(seed)
    nodes = {i for i in range(n * n)}
    edges = {}

    for u in range(n * n):
        row, col = divmod(u, n)
        if col < n - 1:
            v = u + 1
            weight = rng.randint(1, 20)
            edges[(u, v)] = {
                "weight": weight
            }
            if not directed:
                edges[(v, u)] = {
                    "weight": weight
                }
        if row < n - 1:
            v = u + n
            weight = rng.randint(1, 20)
            edges[(u, v)] = {
                "weight": weight
            }
            if not directed:
                edges[(v, u)] = {
                    "weight": weight
                }

    return nodes, edges


def write(nodes, edges, outdir):
    """TO-DO: add function doc here"""
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "node.tsv"), "w") as f:
        f.write("node_id\n")
        for v in range(len(nodes)):
            f.write(f"{v}\n")
    with open(os.path.join(outdir, "edges.tsv"), "w") as f:
            f.write("node_id_from\tnode_id_to\tweight\n")
            for (u, v) in edges:
                f.write(f"{u}\t{v}\t{edges[(u, v)]['weight']}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--directed", action="store_true")
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--type", required=True, choices=("erdos_renyi", "cycle", "complete", "path", "random_trees", "balanced_trees"),
                            help="query to run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.type == "erdos_renyi":
        nodes, edges = erdos_renyi(a.nodes, 0.1, directed=a.directed, seed=a.seed)
    elif a.type == "cycle":
        nodes, edges = cycle(a.nodes, directed=a.directed, seed=a.seed)
    elif a.type == "complete":
        nodes, edges = complete_graph(a.nodes, directed=a.directed, seed=a.seed)
    elif a.type == "path":
        nodes, edges = path_graph(a.nodes, directed=a.directed, seed=a.seed)
    elif a.type == "random_trees":
        nodes, edges = random_trees(a.nodes, directed=a.directed, seed=a.seed)
    elif a.type == "balanced_trees":
        nodes, edges = balanced_trees(a.nodes, a.degree, directed=a.directed, seed=a.seed)
    else:
        raise ValueError(f"Unknown graph type: {a.type}")

    write(nodes, edges, a.out)