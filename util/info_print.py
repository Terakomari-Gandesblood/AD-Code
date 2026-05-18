from typing import Tuple

import torch
from torch_geometric.data import HeteroData


def summarize_hetero_graph(data: HeteroData) -> None:
    SINGLETONS = ('ego', 'gnss', 'traffic_light', 'lane')
    """
    对 build_hetero_graph 的输出进行快速体检：
    - 节点计数（含 vehicle 数量）
    - 单节点类型的出/入度（跨所有边类型累加）
    - 车辆节点相关的 KNN/连接统计
    """
    print("==== Hetero Graph Summary ====")

    # 1) 节点统计
    print("Node types:", data.node_types)
    for ntype in data.node_types:
        num = data[ntype].x.size(0) if 'x' in data[ntype] else 0
        print(f"  - {ntype:13s}: {num} nodes")

    # 2) 单节点类型（ego/gnss/imu/traffic_light/lane）的出/入度
    #    单节点索引恒为 0；统计跨所有边类型的总出入度
    def single_node_degrees(_ntype: str) -> Tuple[int, int]:
        _out_deg = 0
        _in_deg = 0
        for _e_type in data.edge_types:
            src_t, _, dst_t = _e_type
            _ei = data[_e_type].edge_index if 'edge_index' in data[_e_type] else None
            if _ei is None or _ei.numel() == 0:
                continue
            if src_t == _ntype:
                _out_deg += int((_ei[0] == 0).sum().item())
            if dst_t == _ntype:
                _in_deg += int((_ei[1] == 0).sum().item())
        return _out_deg, _in_deg

    print("\nSingleton node in/out degrees (sum over all relations):")
    for ntype in SINGLETONS:
        if ntype in data.node_types:
            out_d, in_d = single_node_degrees(ntype)
            print(f"  - {ntype:13s}: out={out_d:3d}, in={in_d:3d}")
        else:
            print(f"  - {ntype:13s}: (absent)")

    # 3) 车辆节点相关统计
    if 'vehicle' in data.node_types and 'x' in data['vehicle']:
        N = data['vehicle'].x.size(0)
        print(f"\nVehicles: N = {N}")

        # ego -> vehicle
        e2v_edges = 0
        if ('ego', 'to', 'vehicle') in data.edge_types and 'edge_index' in data[('ego', 'to', 'vehicle')]:
            e2v_edges = data[('ego', 'to', 'vehicle')].edge_index.size(1)
        print(f"  ego -> vehicle edges: {e2v_edges}")

        # vehicle -> ego
        v2e_edges = 0
        if ('vehicle', 'to', 'ego') in data.edge_types and 'edge_index' in data[('vehicle', 'to', 'ego')]:
            v2e_edges = data[('vehicle', 'to', 'ego')].edge_index.size(1)
        print(f"  vehicle -> ego edges: {v2e_edges}")

        # vehicle -> vehicle (KNN)
        if ('vehicle', 'to', 'vehicle') in data.edge_types and 'edge_index' in data[('vehicle', 'to', 'vehicle')]:
            ei_vv = data[('vehicle', 'to', 'vehicle')].edge_index
            v2v_edges = ei_vv.size(1)
            # 统计每个车辆节点的出/入度
            if v2v_edges > 0:
                src, dst = ei_vv[0], ei_vv[1]  # (2, E)
                out_deg = torch.bincount(src, minlength=N)
                in_deg = torch.bincount(dst, minlength=N)
                out_avg = float(out_deg.float().mean().item())
                in_avg = float(in_deg.float().mean().item())
                out_max = int(out_deg.max().item())
                in_max = int(in_deg.max().item())
                print(f"  vehicle -> vehicle edges: {v2v_edges}")
                print(f"    out-degree avg={out_avg:.2f}, max={out_max}")
                print(f"    in-degree  avg={in_avg:.2f},  max={in_max}")
            else:
                print(f"  vehicle -> vehicle edges: {v2v_edges}")
        else:
            print("  vehicle -> vehicle edges: 0")
    else:
        print("\nVehicles: N = 0 (no vehicle nodes)")

    # 4) 可选：列出每种边类型的边数
    print("\nEdge types and counts:")
    for e_type in data.edge_types:
        ei = data[e_type].edge_index if 'edge_index' in data[e_type] else None
        cnt = int(ei.size(1)) if (ei is not None) else 0
        print(f"  - {e_type}: {cnt}")

    print("==== End Summary ====")
