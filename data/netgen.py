#!/usr/bin/env python3
"""
COSC 4367 Project 2 — road-network generator.

Produces nodes.tsv / edges.tsv for the "Model It as a Network" project.

The graph is deliberately ROAD-LIKE rather than random-sparse:
  * a perturbed grid backbone (near planar, degree ~3-4, diameter ~sqrt(V))
  * random edge removal for irregularity and dead ends
  * a RIVER cutting the map, crossed by only a few BRIDGES
  * a handful of long "highway" edges, confined to one side of the river
  * shelters biased to the east bank, zones biased to the west bank

The river is the point: it guarantees that the `critical` query has a dramatic
answer (close a bridge and the worst evacuation time explodes), and in Phase 2
the min cut lands on exactly those bridges.

Usage:
    python3 netgen.py --nodes 200 --seed 7 --out instances/small
"""

import argparse, heapq, math, os, random, sys
from collections import deque

LOCAL, HIGHWAY, BRIDGE = "local", "highway", "bridge"

STREETS = ["Oak", "Maple", "Cedar", "Pine", "Elm", "Walnut", "Birch", "Ash",
           "Willow", "Spruce", "Chestnut", "Poplar", "Hickory", "Laurel"]
SHELTER_NAMES = ["Jefferson_Elem", "Riverside_HS", "Civic_Center", "Fairgrounds",
                 "St_Marks_Hall", "Northside_Gym", "Community_College", "Armory",
                 "Lincoln_Middle", "Convention_Hall"]
ZONE_NAMES = ["Riverside", "Millbrook", "Eastgate", "Harbor_Flats", "Old_Town",
              "Westfield", "Cedar_Park", "Lowland", "Brickyard", "Ferry_Point",
              "Draymoor", "Kilnside", "Northbank", "Southmead", "Greenhollow",
              "Tanners_Row", "Quarry_End", "Saltmarsh"]


def two_edge_connected_core(V, edges):
    """Largest component that survives removal of every cut edge.

    Zones placed here cannot be stranded by a single road closure, so the
    `critical` query reports a genuine detour rather than a severed cul-de-sac.
    Returns None when the core spans fewer than half the nodes (e.g. a
    single-bridge map), in which case the caller falls back to a degree test.
    """
    adj = {v: [] for v in range(V)}
    for i, (a, b) in enumerate(sorted(edges)):
        adj[a].append((b, i)); adj[b].append((a, i))

    disc, low = {}, {}
    cut = set()
    timer = 0
    for root in range(V):
        if root in disc:
            continue
        stack = [(root, -1, iter(adj[root]))]
        disc[root] = low[root] = timer; timer += 1
        while stack:
            x, pe, it = stack[-1]
            advanced = False
            for (y, ei) in it:
                if ei == pe:
                    continue
                if y in disc:
                    low[x] = min(low[x], disc[y])
                else:
                    disc[y] = low[y] = timer; timer += 1
                    stack.append((y, ei, iter(adj[y])))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                if stack:
                    px = stack[-1][0]
                    low[px] = min(low[px], low[x])
                    if low[x] > disc[px]:
                        cut.add(pe)

    seen, best = set(), []
    for s in range(V):
        if s in seen:
            continue
        q, comp = deque([s]), []
        seen.add(s)
        while q:
            x = q.popleft(); comp.append(x)
            for (y, ei) in adj[x]:
                if ei in cut or y in seen:
                    continue
                seen.add(y); q.append(y)
        if len(comp) > len(best):
            best = comp
    return set(best) if len(best) >= V * 0.5 else None


def build(n_target, seed, n_bridges=3, del_frac=0.18, oneway_frac=0.08, extra=0,
          speeds=(14, 18, 76), n_shelter=None, n_zone=None):
    rng = random.Random(seed)

    # ---- grid dimensions: a bit wider than tall, like a city ----
    W = max(3, int(round(math.sqrt(n_target * 1.30))))
    H = max(3, int(round(n_target / W)))
    V = W * H

    def nid(c, r):
        return r * W + c

    # ---- jittered positions ----
    pos = {}
    for r in range(H):
        for c in range(W):
            pos[nid(c, r)] = (c * 100 + rng.uniform(-22, 22),
                              r * 100 + rng.uniform(-22, 22))

    # ---- grid edges ----
    edges = {}          # (u,v) undirected key u<v  ->  dict
    for r in range(H):
        for c in range(W):
            u = nid(c, r)
            if c + 1 < W:
                edges[(u, nid(c + 1, r))] = {"kind": LOCAL}
            if r + 1 < H:
                edges[(u, nid(c, r + 1))] = {"kind": LOCAL}

    # ---- the river: a vertical cut between columns xr-1 and xr ----
    xr = W // 2
    crossing = [k for k in edges if (k[0] % W) < xr <= (k[1] % W)
                or (k[1] % W) < xr <= (k[0] % W)]
    # Space the bridges evenly down the river instead of picking at random.
    # Two bridges at opposite ends means closing one forces a detour of roughly
    # half the map, which is what gives `critical` a dramatic answer.
    by_row = {}
    for k in crossing:
        by_row[min(k) // W] = k
    rows = sorted(by_row)
    n_bridges = max(1, min(n_bridges, len(rows)))
    picks = [rows[int(round((i + 0.5) * len(rows) / n_bridges)) % len(rows)]
             for i in range(n_bridges)]
    bridges = {by_row[r] for r in dict.fromkeys(picks)}
    for k in crossing:
        if k not in bridges:
            del edges[k]
    for k in bridges:
        edges[k]["kind"] = BRIDGE

    # ---- random deletion for irregularity, never a bridge ----
    deletable = [k for k in edges if k not in bridges]
    rng.shuffle(deletable)
    for k in deletable[:int(len(deletable) * del_frac)]:
        del edges[k]

    # ---- repair: make the undirected graph connected again ----
    def components():
        adj = {v: [] for v in range(V)}
        for (a, b) in edges:
            adj[a].append(b); adj[b].append(a)
        seen, comps = set(), []
        for s in range(V):
            if s in seen:
                continue
            q, comp = deque([s]), []
            seen.add(s)
            while q:
                x = q.popleft(); comp.append(x)
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y); q.append(y)
            comps.append(comp)
        return comps

    comps = components()
    while len(comps) > 1:
        comps.sort(key=len, reverse=True)
        main = set(comps[0])
        other = comps[1]
        # reconnect on the same bank so the river stays the only crossing
        best = None
        for a in other:
            for b in main:
                if (a % W < xr) != (b % W < xr):
                    continue
                d = math.dist(pos[a], pos[b])
                if best is None or d < best[0]:
                    best = (d, a, b)
        if best is None:                      # whole component stranded: bridge it
            a, b = other[0], comps[0][0]
        else:
            _, a, b = best
        key = (min(a, b), max(a, b))
        edges[key] = {"kind": BRIDGE if (a % W < xr) != (b % W < xr) else LOCAL}
        if edges[key]["kind"] == BRIDGE:
            bridges.add(key)
        comps = components()

    # ---- highways: long edges, confined to one bank ----
    n_hw = max(1, V // 60)
    tries = 0
    while n_hw > 0 and tries < 400:
        tries += 1
        a, b = rng.randrange(V), rng.randrange(V)
        if a == b:
            continue
        if (a % W < xr) != (b % W < xr):       # must not cross the river
            continue
        if math.dist(pos[a], pos[b]) < 250:
            continue
        key = (min(a, b), max(a, b))
        if key in edges:
            continue
        edges[key] = {"kind": HIGHWAY}
        n_hw -= 1

    # ---- optional densification for the 3.3 sparsity sweep ----
    # These chords are NOT road-like and may cross the river; a densified
    # instance is for timing studies only, never for the `critical` query.
    tries = 0
    while extra > 0 and tries < extra * 60:
        tries += 1
        a, b = rng.randrange(V), rng.randrange(V)
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in edges:
            continue
        edges[key] = {"kind": LOCAL}
        extra -= 1

    # ---- weights, capacities, direction ----
    # speeds = (local, bridge, highway) as distance-per-minute divisors.
    # The defaults give ~4x the integer resolution of a naive 55/70/300 scale,
    # which removes every tied nearest-shelter from the shipped instances while
    # preserving the speed ratios (bridge 1.29x local, highway 5.43x local).
    for (a, b), e in edges.items():
        d = math.dist(pos[a], pos[b])
        if e["kind"] == HIGHWAY:
            e["minutes"] = max(1, int(round(d / speeds[2])))
            e["capacity"] = rng.randrange(300, 520, 10)
            e["oneway"] = 0
        elif e["kind"] == BRIDGE:
            e["minutes"] = max(1, int(round(d / speeds[1])))
            e["capacity"] = rng.randrange(60, 130, 10)
            e["oneway"] = 0
        else:
            e["minutes"] = max(1, int(round(d / speeds[0])))
            e["capacity"] = rng.randrange(40, 100, 10)
            e["oneway"] = 1 if rng.random() < oneway_frac else 0

    # ---- node roles: shelters east, zones west ----
    # Zones and shelters go on nodes with at least two incident roads; a zone
    # on a dead end is disconnected by a single closure, which drowns out the
    # bridge structure that `critical` is supposed to reveal.
    deg = {v: 0 for v in range(V)}
    for (a, b) in edges:
        deg[a] += 1; deg[b] += 1

    core = two_edge_connected_core(V, edges)

    def banks(pred):
        e = [v for v in range(V) if v % W >= xr and pred(v)]
        w = [v for v in range(V) if v % W < xr and pred(v)]
        return e, w

    east, west = ([], [])
    if core:
        east, west = banks(lambda v: v in core)
    if len(east) < 3 or len(west) < 6:          # not enough sites in the core
        east, west = banks(lambda v: deg[v] >= 2)
    rng.shuffle(east); rng.shuffle(west)

    # Defaults scale sublinearly with V; --shelters/--zones override them.
    # The name lists no longer cap these (see pick() above).
    if n_shelter is None:
        n_shelter = max(2, min(10, round(V ** 0.38)))
    if n_zone is None:
        n_zone = max(3, min(18, round(V ** 0.52)))
    n_shelter = min(n_shelter, max(1, len(east) - 1))
    n_zone = min(n_zone, max(1, len(west) + len(east) - n_shelter - 1))

    # Shelters sit on the far bank (high ground); zones are mostly on the near
    # bank.  Every near-bank evacuation therefore has to cross a bridge, which
    # is what makes `critical` dramatic and what the Phase 2 min cut will find.
    shelters = east[:n_shelter]
    n_east_zone = max(1, n_zone // 7)
    zones = [v for v in west if v not in shelters][:n_zone - n_east_zone]
    zones += [v for v in east[n_shelter:] if v not in shelters][:n_east_zone]

    kind = {v: "junction" for v in range(V)}
    for v in shelters:
        kind[v] = "shelter"
    for v in zones:
        kind[v] = "zone"

    pass_time = {}
    for v in range(V):
        if kind[v] == "junction":
            pass_time[v] = 0 if rng.random() < 0.78 else rng.randint(1, 6)
        else:
            pass_time[v] = 0

    # On a small graph the 22% draw above can leave every junction at zero,
    # which would make the instance useless for checking the node-cost
    # transformation of handout 3.2: a broken splitter passes it. Enforce a
    # floor. The floor is below the expected count for any V of real size, so
    # this only ever fires on the tiny instances. Keep it after the draw so
    # larger instances are bit-identical to before this guard existed.
    junctions = [v for v in range(V) if kind[v] == "junction"]
    if junctions:
        floor = max(2, round(0.10 * len(junctions)))
        floor = min(floor, len(junctions))
        hot = [v for v in junctions if pass_time[v] > 0]
        if len(hot) < floor:
            cold = [v for v in junctions if pass_time[v] == 0]
            for v in rng.sample(cold, floor - len(hot)):
                pass_time[v] = rng.randint(1, 6)

    def pick(lst, i):
        return lst[i] if i < len(lst) else f"{lst[i % len(lst)]}_{i // len(lst) + 1}"

    # ---- directed arcs; repair any zone that cannot reach a shelter ----
    def arcs():
        out = []
        for (a, b), e in edges.items():
            out.append((a, b, e))
            if not e["oneway"]:
                out.append((b, a, e))
        return out

    def unreachable_zones():
        radj = {v: [] for v in range(V)}
        for (a, b, e) in arcs():
            radj[b].append(a)                 # predecessor lists
        seen = set(shelters)
        q = deque(shelters)
        while q:
            x = q.popleft()
            for y in radj[x]:
                if y not in seen:
                    seen.add(y); q.append(y)
        return [z for z in zones if z not in seen]

    guard = 0
    while True:
        bad = unreachable_zones()
        if not bad or guard > 200:
            break
        guard += 1
        for (k, e) in edges.items():          # relax one-ways until everyone connects
            if e["oneway"]:
                e["oneway"] = 0
                break
        else:
            break

    # ---- directed robustness: relocate zones a single closure can strand ----
    # A zone can only be stranded by removing an edge that lies on every route
    # to a shelter, so it is enough to test the edges on its own path.  The
    # undirected core test cannot catch this because one-way edges break
    # DIRECTED reachability.
    spare = [v for v in west + east if v not in shelters and v not in zones]
    shelterset = set(shelters)
    for _round in range(6):
        fadj = {v: [] for v in range(V)}
        radj = {v: [] for v in range(V)}
        for (a, b, e) in arcs():
            fadj[a].append(b); radj[b].append(a)

        parent = {s: None for s in shelters}      # BFS tree over predecessors
        q = deque(shelters)
        while q:
            x = q.popleft()
            for y in radj[x]:
                if y not in parent:
                    parent[y] = x; q.append(y)

        def reaches(z, skip):
            seen = {z}; q2 = deque([z])
            while q2:
                x = q2.popleft()
                if x in shelterset:
                    return True
                for y in fadj[x]:
                    if y in seen or (min(x, y), max(x, y)) == skip:
                        continue
                    seen.add(y); q2.append(y)
            return False

        fragile = []
        for z in zones:
            path, x = [], z
            while parent.get(x) is not None:
                nxt = parent[x]
                path.append((min(x, nxt), max(x, nxt)))
                x = nxt
            if x not in shelterset or any(not reaches(z, k) for k in path):
                fragile.append(z)
        if not fragile:
            break
        for z in fragile:
            if not spare:
                break
            repl = spare.pop()
            zones[zones.index(z)] = repl
            # `kind` is what write() emits, so it MUST move with `zones`.
            # Without this the relocation lives only in the local list and
            # never reaches nodes.tsv, silently making this whole pass a
            # no-op in the generated files.
            kind[z] = "junction"
            kind[repl] = "zone"

    # Relocating a zone can land it on a junction that already drew a
    # pass_time, which would ship a zone with a nonzero node cost and break
    # the documented convention that only junctions are signalized.  It does
    # not change any answer (a trip never pays its own endpoints' costs), but
    # it is inconsistent data that students would reasonably trip over.
    for v in range(V):
        if kind[v] != "junction":
            pass_time[v] = 0

    # ---- make the instance SENSITIVE to node costs ----
    # An instance is useless for checking the 3.2 node-cost transformation
    # unless ignoring pass_time changes at least one zone's answer.  A count
    # floor is not enough: the costly junctions have to sit ON the evacuation
    # routes.  So find the routes with pass_time ignored, and if no interior
    # junction on any of them is costly, make some of them costly.
    #
    # On any instance of real size this condition is already satisfied (the
    # routes are long and ~22% of junctions are costly), so this fires only on
    # the small hand-checkable instances and leaves the others untouched.
    def route_interiors():
        """Interior junctions of each zone's cheapest route, pass_time ignored.
        (Generator-internal placement check. This is NOT a model answer for the
        handout's modeling questions -- it solves a different problem, ignores
        node costs, and your memo must justify your own approach regardless.)"""
        radj = {v: [] for v in range(V)}
        for (a, b, e) in arcs():
            radj[b].append((a, e["minutes"]))     # weighted predecessor lists
        INF = float("inf")
        dist = {v: INF for v in range(V)}
        parent = {}
        pq = []
        for s in shelters:
            dist[s] = 0; parent[s] = None; pq.append((0, s))
        heapq.heapify(pq)
        done = set()
        while pq:
            d, x = heapq.heappop(pq)
            if x in done:
                continue
            done.add(x)
            for (y, w) in radj[x]:
                if d + w < dist[y]:
                    dist[y] = d + w; parent[y] = x
                    heapq.heappush(pq, (dist[y], y))
        interiors = set()
        for z in zones:
            if dist[z] == INF:
                continue
            x = parent.get(z)                     # skip the zone itself
            while x is not None and parent.get(x) is not None:
                if kind[x] == "junction":
                    interiors.add(x)              # skip the shelter (parent None)
                x = parent[x]
        return interiors

    interiors = route_interiors()
    if interiors and not any(pass_time[v] > 0 for v in interiors):
        pick_n = max(2, round(0.25 * len(interiors)))
        for v in rng.sample(sorted(interiors), min(pick_n, len(interiors))):
            pass_time[v] = rng.randint(1, 6)

    # ---- names, assigned LAST ----
    # This has to come after the robustness pass, which relocates zones. Name
    # them any earlier and a relocated zone keeps the street-style name it had
    # while it was still a junction, so nodes.tsv ships a zone called
    # "Ash_7" sitting among Riverside and Millbrook. Harmless to every answer,
    # but it reads like a mislabelled node in the one instance students study
    # by hand. Same root cause as the `kind` and `pass_time` staleness above:
    # anything derived from `kind` must be computed after `kind` settles.
    name = {}
    si = zi = 0
    for v in range(V):
        if kind[v] == "shelter":
            name[v] = pick(SHELTER_NAMES, si)
            si += 1
        elif kind[v] == "zone":
            name[v] = pick(ZONE_NAMES, zi)
            zi += 1
        else:
            name[v] = f"{STREETS[v % len(STREETS)]}_{v}"

    meta = dict(W=W, H=H, V=V, xr=xr, bridges=sorted(bridges),
                shelters=sorted(shelters), zones=sorted(zones),
                n_edges=len(edges))
    return dict(V=V, pos=pos, edges=edges, kind=kind, name=name,
                pass_time=pass_time, shelters=sorted(shelters),
                zones=sorted(zones), bridges=bridges, meta=meta)


def write(net, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "nodes.tsv"), "w") as f:
        f.write("id\tname\tkind\tpass_time\n")
        for v in range(net["V"]):
            f.write(f"{v}\t{net['name'][v]}\t{net['kind'][v]}\t{net['pass_time'][v]}\n")
    with open(os.path.join(outdir, "edges.tsv"), "w") as f:
        f.write("u\tv\tminutes\tcapacity\toneway\n")
        for (a, b) in sorted(net["edges"]):
            e = net["edges"][(a, b)]
            f.write(f"{a}\t{b}\t{e['minutes']}\t{e['capacity']}\t{e['oneway']}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bridges", type=int, default=3)
    ap.add_argument("--extra", type=int, default=0,
                    help="extra random chords; densifies for the 3.3 sparsity sweep")
    ap.add_argument("--shelters", type=int, default=None)
    ap.add_argument("--zones", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    net = build(a.nodes, a.seed, a.bridges, extra=a.extra,
                n_shelter=a.shelters, n_zone=a.zones)
    write(net, a.out)
    m = net["meta"]
    print(f"{a.out}: V={m['V']} ({m['W']}x{m['H']})  E={m['n_edges']}  "
          f"bridges={len(m['bridges'])}  shelters={len(m['shelters'])}  zones={len(m['zones'])}")
