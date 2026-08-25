from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
import json
import math
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, eye
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from .topology import TopologyData, infer_num_object_points, load_topology, normalize_score

EPS = 1.0e-12


@dataclass
class CoarseningResult:
    topology_path: Path
    metadata: dict


def _object_geometry(data: TopologyData):
    n = infer_num_object_points(data)
    points = np.asarray(data.points_full[:n], dtype=np.float64)
    edges = np.asarray(data.object_springs, dtype=np.int64)
    rest = np.asarray(data.object_rest_lengths, dtype=np.float64)
    y = np.asarray(data.object_spring_Y, dtype=np.float64)
    p0 = points[edges[:, 0]]
    p1 = points[edges[:, 1]]
    vec = p0 - p1
    length = np.linalg.norm(vec, axis=1)
    safe = np.where(rest > EPS, rest, np.maximum(length, EPS))
    direction = vec / safe[:, None]
    axial_k = y / np.maximum(rest, EPS)
    return n, points, edges, rest, y, direction, axial_k


def _remove_translation(x: np.ndarray, masses: np.ndarray | None = None):
    if masses is None:
        return x - x.mean(axis=0, keepdims=True)
    w = masses / max(float(masses.sum()), EPS)
    return x - np.sum(w[:, None] * x, axis=0, keepdims=True)


def _apply_K(x, edges, direction, axial_k, n):
    out = np.zeros((n, 3), dtype=np.float64)
    i, j = edges[:, 0], edges[:, 1]
    delta = x[i] - x[j]
    stretch = np.sum(delta * direction, axis=1)
    force = axial_k[:, None] * stretch[:, None] * direction
    np.add.at(out, i, force)
    np.add.at(out, j, -force)
    return out


def _mgs(w, basis):
    for _ in range(2):
        for v in basis:
            w = w - float(np.sum(w * v)) * v
    return w


def build_trajectory_signature(
    inference_path: str | Path,
    data: TopologyData,
    start: int,
    end: int,
    rank: int = 8,
):
    """Low-rank trajectory displacement signature, useful as an ablation."""
    n = infer_num_object_points(data)
    with Path(inference_path).open("rb") as f:
        traj = np.asarray(pickle.load(f), dtype=np.float64)
    end = min(int(end), len(traj))
    start = max(0, int(start))
    x = traj[start:end, :n] - data.points_full[:n][None]
    if len(x) == 0:
        raise ValueError("Empty frame range for trajectory signature")
    X = np.transpose(x, (1, 2, 0)).reshape(3*n, len(x))
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    r = min(rank, U.shape[1])
    return U[:, :r].reshape(n, 3, r)


def build_krylov_signature(
    inference_path: str | Path,
    data: TopologyData,
    start: int,
    end: int,
    rank: int = 10,
    operator: str = "MinvK",
):
    n, points, edges, rest, y, direction, axial_k = _object_geometry(data)
    masses = np.asarray(data.masses[:n], dtype=np.float64)
    with Path(inference_path).open("rb") as f:
        traj = np.asarray(pickle.load(f), dtype=np.float64)
    end = min(int(end), len(traj))
    start = max(0, int(start))
    if end - start < 2:
        raise ValueError("Need at least two frames for Krylov seed")
    b = traj[end-1, :n] - traj[start, :n]
    basis = []
    w = _remove_translation(b, masses if operator == "MinvK" else None)
    norm = np.linalg.norm(w)
    if norm < EPS:
        raise RuntimeError("Near-zero Krylov seed")
    basis.append(w / norm)
    for _ in range(1, rank):
        w = _apply_K(basis[-1], edges, direction, axial_k, n)
        if operator == "MinvK":
            w = w / np.maximum(masses[:, None], EPS)
        elif operator != "K":
            raise ValueError(operator)
        w = _remove_translation(w, masses if operator == "MinvK" else None)
        w = _mgs(w, basis)
        norm = np.linalg.norm(w)
        if norm < EPS:
            break
        basis.append(w / norm)
    V = np.stack(basis, axis=0)
    return np.transpose(V, (1, 2, 0))


def _assemble_edge_operator(edges, direction, coeff, n):
    rows, cols, vals = [], [], []
    for (i, j), d, c in zip(edges, direction, coeff):
        block = float(c) * np.outer(d, d)
        for a in range(3):
            for b in range(3):
                ii, jj = 3*int(i)+a, 3*int(j)+a
                ci, cj = 3*int(i)+b, 3*int(j)+b
                v = float(block[a, b])
                rows += [ii, jj, ii, jj]
                cols += [ci, cj, cj, ci]
                vals += [v, v, -v, -v]
    return coo_matrix((vals, (rows, cols)), shape=(3*n, 3*n)).tocsc()


def build_soar_signature(
    inference_path: str | Path,
    data: TopologyData,
    start: int,
    end: int,
    rank: int = 5,
    dashpot: float = 100.0,
    drag: float = 3.0,
    reg_scale: float = 1.0e-6,
):
    """SOAR-inspired M-D-K signature on an object-only linearized surrogate."""
    n, points, edges, rest, y, direction, axial_k = _object_geometry(data)
    masses = np.asarray(data.masses[:n], dtype=np.float64)
    with Path(inference_path).open("rb") as f:
        traj = np.asarray(pickle.load(f), dtype=np.float64)
    end = min(int(end), len(traj))
    start = max(0, int(start))
    if end - start < 2:
        raise ValueError("Need at least two frames for SOAR seed")
    b = traj[end-1, :n] - traj[start, :n]

    K = _assemble_edge_operator(edges, direction, axial_k, n)
    reg = reg_scale * max(float(np.mean(axial_k)), EPS)
    K = K + reg * eye(3*n, format="csc")

    D = _assemble_edge_operator(
        edges, direction, np.full(len(edges), float(dashpot)), n
    )
    D = D + float(drag) * eye(3*n, format="csc")
    Mdiag = np.repeat(np.maximum(masses, EPS), 3)
    M = csc_matrix((Mdiag, (np.arange(3*n), np.arange(3*n))), shape=(3*n, 3*n))

    lu = splu(K)
    basis = []

    def clean(v):
        v = v.reshape(n, 3)
        v = _remove_translation(v, masses)
        v = _mgs(v, basis)
        norm = np.linalg.norm(v)
        return v, norm

    v1 = lu.solve(b.reshape(-1)).reshape(n, 3)
    v1, norm = clean(v1)
    if norm < EPS:
        raise RuntimeError("Near-zero SOAR v1")
    basis.append(v1 / norm)

    if rank >= 2:
        rhs = -(D @ basis[-1].reshape(-1))
        v2 = lu.solve(rhs).reshape(n, 3)
        v2, norm = clean(v2)
        if norm >= EPS:
            basis.append(v2 / norm)

    while len(basis) < rank and len(basis) >= 2:
        rhs = -(D @ basis[-1].reshape(-1) + M @ basis[-2].reshape(-1))
        v = lu.solve(rhs).reshape(n, 3)
        v, norm = clean(v)
        if norm < EPS:
            break
        basis.append(v / norm)

    V = np.stack(basis, axis=0)
    return np.transpose(V, (1, 2, 0))


def _controller_masks(data: TopologyData, n_obj: int):
    masks = [0] * n_obj
    for edge in np.asarray(data.controller_springs, dtype=np.int64):
        a, b = int(edge[0]), int(edge[1])
        if a < n_obj <= b:
            obj, ctrl = a, b - n_obj
        elif b < n_obj <= a:
            obj, ctrl = b, a - n_obj
        else:
            continue
        masks[obj] |= 1 << ctrl
    return masks


def _node_importance(signature):
    return normalize_score(np.sum(np.asarray(signature, dtype=np.float64)**2, axis=(1,2)))


def agglomerative_clusters(
    points,
    edges,
    signature,
    node_importance,
    target_nodes,
    *,
    alpha_dyn=1.0,
    beta_geo=1.0,
    protect_top_pct=10.0,
    max_cluster_size=16,
    controller_masks=None,
):
    n = len(points)
    controller_masks = controller_masks or [0]*n
    threshold = (
        np.percentile(node_importance, 100.0-protect_top_pct)
        if protect_top_pct > 0 else np.inf
    )
    protected = node_importance >= threshold

    parent = np.arange(n)
    active = np.ones(n, dtype=bool)
    mass = np.ones(n, dtype=np.float64)
    size = np.ones(n, dtype=np.int64)
    centroid = np.asarray(points, dtype=np.float64).copy()
    sig = np.asarray(signature, dtype=np.float64).copy()
    prot = protected.copy()
    cmask = [int(x) for x in controller_masks]
    adj = [set() for _ in range(n)]
    for i,j in edges:
        i,j=int(i),int(j)
        adj[i].add(j); adj[j].add(i)

    edge_len = np.linalg.norm(points[edges[:,0]] - points[edges[:,1]], axis=1)
    geo_scale = max(float(np.median(edge_len))**2, EPS)
    vi,vj=sig[edges[:,0]],sig[edges[:,1]]
    dyn = np.sum((vi-vj)**2, axis=(1,2)) / (
        np.sum(vi*vi, axis=(1,2))+np.sum(vj*vj, axis=(1,2))+EPS
    )
    dyn_pos = dyn[dyn > EPS]
    dyn_scale = max(float(np.median(dyn_pos)) if len(dyn_pos) else 1.0, EPS)

    version = np.zeros(n, dtype=np.int64)
    heap=[]

    def allowed(a,b):
        if not(active[a] and active[b]) or b not in adj[a]:
            return False
        if max_cluster_size and size[a]+size[b] > max_cluster_size:
            return False
        if prot[a] and prot[b]:
            return False
        if cmask[a] and cmask[b] and cmask[a] != cmask[b]:
            return False
        return True

    def cost(a,b):
        g = np.sum((centroid[a]-centroid[b])**2)/geo_scale
        num=np.sum((sig[a]-sig[b])**2)
        den=np.sum(sig[a]**2)+np.sum(sig[b]**2)+EPS
        d=(num/den)/dyn_scale
        return alpha_dyn*d + beta_geo*g

    def push(a,b):
        if a>b: a,b=b,a
        if allowed(a,b):
            heapq.heappush(heap,(cost(a,b),a,b,int(version[a]),int(version[b])))

    for a in range(n):
        for b in adj[a]:
            if a<b: push(a,b)

    active_count=n
    while active_count > target_nodes:
        chosen=None
        while heap:
            c,a,b,va,vb=heapq.heappop(heap)
            if not active[a] or not active[b] or va!=version[a] or vb!=version[b] or b not in adj[a]:
                continue
            if not allowed(a,b):
                continue
            fresh=cost(a,b)
            if not math.isclose(c,fresh,rel_tol=1e-10,abs_tol=1e-12):
                heapq.heappush(heap,(fresh,a,b,int(version[a]),int(version[b])))
                continue
            chosen=(a,b); break
        if chosen is None:
            raise RuntimeError(
                f"Coarsening stopped at {active_count}, target={target_nodes}. "
                "Relax protection/max_cluster_size."
            )
        a,b=chosen
        if a>b: a,b=b,a
        mt=mass[a]+mass[b]
        centroid[a]=(mass[a]*centroid[a]+mass[b]*centroid[b])/mt
        sig[a]=(mass[a]*sig[a]+mass[b]*sig[b])/mt
        mass[a]=mt
        size[a]+=size[b]
        prot[a]=prot[a] or prot[b]
        cmask[a]|=cmask[b]
        parent[b]=a
        active[b]=False
        version[a]+=1; version[b]+=1
        neigh=(adj[a]|adj[b])-{a,b}
        adj[a]=set()
        for nb in neigh:
            if not active[nb]: continue
            adj[nb].discard(a); adj[nb].discard(b)
            adj[nb].add(a); adj[a].add(nb)
        adj[b].clear()
        for nb in list(adj[a]): push(a,nb)
        active_count-=1

    def find(x):
        while parent[x] != x:
            parent[x] = parent[int(parent[x])]
            x = int(parent[x])
        return x

    roots=np.asarray([find(i) for i in range(n)],dtype=np.int64)
    active_roots=np.where(active)[0]
    lut={int(r):k for k,r in enumerate(active_roots)}
    cluster=np.asarray([lut[int(r)] for r in roots],dtype=np.int64)
    reduced_points=centroid[active_roots]
    reduced_masses=np.zeros(len(active_roots),dtype=np.float64)
    for i,c in enumerate(cluster):
        reduced_masses[c]+=1.0
    return cluster, reduced_points, reduced_masses, protected


def _contract_topology(data, cluster, reduced_points):
    n_obj = infer_num_object_points(data)
    n_red = len(reduced_points)
    controllers = np.asarray(data.points_full[n_obj:], dtype=np.float64)
    controller_masses = np.asarray(data.masses[n_obj:], dtype=np.float64)

    obj_acc=defaultdict(float)
    for (i,j),rest,y in zip(data.object_springs,data.object_rest_lengths,data.object_spring_Y):
        a,b=int(cluster[int(i)]),int(cluster[int(j)])
        if a==b: continue
        key=tuple(sorted((a,b)))
        obj_acc[key]+=float(y)/max(float(rest),EPS)

    obj_edges=[]; obj_rest=[]; obj_y=[]
    for (a,b),k in sorted(obj_acc.items()):
        L=float(np.linalg.norm(reduced_points[a]-reduced_points[b]))
        if L <= 1e-4: continue
        obj_edges.append([a,b]); obj_rest.append(L); obj_y.append(k*L)

    ctl_acc=defaultdict(float)
    for (i,j),rest,y in zip(data.controller_springs,data.controller_rest_lengths,data.controller_spring_Y):
        i,j=int(i),int(j)
        if i<n_obj<=j:
            a=int(cluster[i]); b=n_red+(j-n_obj)
        elif j<n_obj<=i:
            a=int(cluster[j]); b=n_red+(i-n_obj)
        else:
            continue
        key=(a,b)
        ctl_acc[key]+=float(y)/max(float(rest),EPS)

    ctl_edges=[]; ctl_rest=[]; ctl_y=[]
    for (a,b),k in sorted(ctl_acc.items()):
        L=float(np.linalg.norm(reduced_points[a]-controllers[b-n_red]))
        if L <= 1e-4: continue
        ctl_edges.append([a,b]); ctl_rest.append(L); ctl_y.append(k*L)

    obj_edges=np.asarray(obj_edges,dtype=np.int64).reshape(-1,2)
    ctl_edges=np.asarray(ctl_edges,dtype=np.int64).reshape(-1,2)
    springs=np.concatenate([obj_edges,ctl_edges],axis=0)
    rest=np.concatenate([np.asarray(obj_rest),np.asarray(ctl_rest)])
    y=np.concatenate([np.asarray(obj_y),np.asarray(ctl_y)])

    red_masses=np.zeros(n_red,dtype=np.float64)
    for old,c in enumerate(cluster):
        red_masses[c]+=float(data.masses[old])
    points_full=np.concatenate([reduced_points,controllers],axis=0)
    masses=np.concatenate([red_masses,controller_masses],axis=0)
    return points_full,springs,rest,masses,y,len(obj_edges)


def build_graph_mapping(original_points, reduced_points, cluster, reduced_object_edges, k=4):
    n_red=len(reduced_points)
    k=min(int(k),n_red)
    adj=[set() for _ in range(n_red)]
    for a,b in reduced_object_edges:
        a,b=int(a),int(b)
        adj[a].add(b); adj[b].add(a)
    tree=cKDTree(reduced_points)
    idx=np.zeros((len(original_points),k),dtype=np.int64)
    w=np.zeros((len(original_points),k),dtype=np.float64)
    for p in range(len(original_points)):
        start=int(cluster[p])
        seen={start}; q=deque([start]); cand=[]
        while q and len(cand)<max(k*4,k):
            u=q.popleft(); cand.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        if len(cand)<k:
            _,fill=tree.query(original_points[p],k=min(n_red,max(k*2,k)))
            for v in np.atleast_1d(fill):
                v=int(v)
                if v not in seen:
                    cand.append(v); seen.add(v)
                if len(cand)>=k: break
        cand=np.asarray(cand,dtype=np.int64)
        d=np.linalg.norm(reduced_points[cand]-original_points[p],axis=1)
        order=np.argsort(d)[:k]
        chosen=cand[order]; dist=d[order]
        raw=1.0/(dist+1e-9)
        if np.any(dist <= 1e-10):
            raw=(dist<=1e-10).astype(np.float64)
        raw/=raw.sum()
        idx[p]=chosen; w[p]=raw
    return idx,w


def generate_hierarchical_node_topology(
    topology_path,
    inference_path,
    output_path,
    *,
    method="soar",
    frame_start=0,
    frame_end=None,
    keep_ratio=0.5,
    signature_rank=None,
    alpha_dyn=1.0,
    beta_geo=1.0,
    protect_top_pct=10.0,
    max_cluster_size=16,
    mapping_k=4,
    dashpot=100.0,
    drag=3.0,
):
    data=load_topology(topology_path)
    n=infer_num_object_points(data)
    with Path(inference_path).open("rb") as f:
        traj=np.asarray(pickle.load(f))
    if frame_end is None: frame_end=len(traj)
    if method=="trajectory":
        sig=build_trajectory_signature(inference_path,data,frame_start,frame_end,signature_rank or 8)
    elif method=="krylov":
        sig=build_krylov_signature(inference_path,data,frame_start,frame_end,signature_rank or 10)
    elif method=="soar":
        sig=build_soar_signature(inference_path,data,frame_start,frame_end,signature_rank or 5,dashpot,drag)
    elif method=="geometry":
        sig=np.zeros((n,3,1),dtype=np.float64)
        alpha_dyn=0.0
    else:
        raise ValueError("method must be trajectory|krylov|soar|geometry")

    imp=_node_importance(sig)
    target=max(1,int(math.ceil(n*float(keep_ratio))))
    cluster,red_points,_,protected=agglomerative_clusters(
        data.points_full[:n],data.object_springs,sig,imp,target,
        alpha_dyn=alpha_dyn,beta_geo=beta_geo,
        protect_top_pct=protect_top_pct,max_cluster_size=max_cluster_size,
        controller_masks=_controller_masks(data,n),
    )
    pts,springs,rest,masses,y,n_obj_spr=_contract_topology(data,cluster,red_points)
    red_obj_edges=springs[:n_obj_spr]
    map_idx,map_w=build_graph_mapping(
        np.asarray(data.points_full[:n],dtype=np.float64),
        red_points,cluster,red_obj_edges,k=mapping_k
    )
    out=Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    meta={
        "method":method,"frame_start":int(frame_start),"frame_end":int(frame_end),
        "original_object_nodes":n,"reduced_object_nodes":len(red_points),
        "keep_ratio_requested":float(keep_ratio),
        "keep_ratio_actual":len(red_points)/n,
        "alpha_dyn":float(alpha_dyn),"beta_geo":float(beta_geo),
        "protect_top_pct":float(protect_top_pct),
        "max_cluster_size":int(max_cluster_size),"mapping_k":int(mapping_k),
        "important_split_rule":"frame_end must not include test frames",
    }
    np.savez_compressed(
        out,points_full=pts,springs=springs,rest_lengths=rest,masses=masses,
        spring_Y=y,num_object_springs=np.asarray(n_obj_spr,dtype=np.int64),
        reduction_type=np.asarray("hierarchical_node_coarsening"),
        node_method=np.asarray(method),object_cluster=cluster,
        original_object_points=np.asarray(data.points_full[:n],dtype=np.float64),
        reduced_object_points=red_points,mapping_indices=map_idx,mapping_weights=map_w,
        protected_nodes=protected,node_signature=sig,node_importance=imp,
        original_num_object_points=np.asarray(n,dtype=np.int64),
        original_num_total_points=np.asarray(len(data.points_full),dtype=np.int64),
        metadata_json=np.asarray(json.dumps(meta)),
    )
    out.with_suffix(".json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return CoarseningResult(out,meta)


def reconstruct_dense_trajectory(reduced_inference_path, topology_path, output_path):
    with Path(reduced_inference_path).open("rb") as f:
        q=np.asarray(pickle.load(f),dtype=np.float64)
    z=np.load(topology_path,allow_pickle=True)
    red0=np.asarray(z["reduced_object_points"],dtype=np.float64)
    dense0=np.asarray(z["original_object_points"],dtype=np.float64)
    idx=np.asarray(z["mapping_indices"],dtype=np.int64)
    w=np.asarray(z["mapping_weights"],dtype=np.float64)
    nred=len(red0)
    dq=q[:,:nred]-red0[None]
    dense=dense0[None]+np.sum(
        w[None,:,:,None]*dq[:,idx,:],axis=2
    )
    controller=q[:,nred:]
    full=np.concatenate([dense,controller],axis=1)
    out=Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(full.astype(np.float32),f)
    return out
