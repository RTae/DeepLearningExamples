#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

import dgl
import torch

from se3_transformer.model import Fiber
from se3_transformer.model.transformer import SE3Transformer


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone SE3Transformer micro-benchmark profiler")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-nodes", type=int, default=64)
    parser.add_argument("--edge-dim", type=int, default=4)
    parser.add_argument("--node-dim", type=int, default=6)
    parser.add_argument("--num-degrees", type=int, default=4)
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=7)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--channels-div", type=int, default=2)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--profile-iters", type=int, default=2)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./profile_out"))
    return parser.parse_args()


def build_dense_graph(num_nodes: int, device: torch.device) -> dgl.DGLGraph:
    src = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes, device=device).repeat(num_nodes)
    keep = src != dst
    graph = dgl.graph((src[keep], dst[keep]), num_nodes=num_nodes, idtype=torch.int32, device=device)

    pos = torch.randn(num_nodes, 3, device=device)
    graph.ndata["pos"] = pos
    edge_src, edge_dst = graph.edges()
    graph.edata["rel_pos"] = pos[edge_dst] - pos[edge_src]
    return graph


def build_batched_inputs(
    batch_size: int,
    num_nodes: int,
    node_dim: int,
    edge_dim: int,
    device: torch.device,
):
    graphs = [build_dense_graph(num_nodes=num_nodes, device=device) for _ in range(batch_size)]
    graph = dgl.batch(graphs)

    num_total_nodes = graph.num_nodes()
    num_total_edges = graph.num_edges()

    # SE3Transformer expects type-0 node/edge features shaped as [count, channels, 1]
    node_feats = {"0": torch.randn(num_total_nodes, node_dim, 1, device=device)}
    edge_feats = {"0": torch.randn(num_total_edges, edge_dim, 1, device=device)}
    return graph, node_feats, edge_feats


def write_table(key_averages, output_dir: Path, filename: str, sort_keys: List[str], row_limit: int = 30):
    table = None
    for sort_key in sort_keys:
        try:
            table = key_averages.table(sort_by=sort_key, row_limit=row_limit)
            break
        except Exception:
            continue

    if table is None:
        table = "Profiler table unavailable for requested sort keys: " + ", ".join(sort_keys)

    with (output_dir / filename).open("w") as f:
        f.write(table)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    use_amp = args.amp and cuda_available

    print(f"Using device: {device}")
    print(f"Output dir: {args.output_dir}")

    major_cc = minor_cc = 0
    if cuda_available:
        major_cc, minor_cc = torch.cuda.get_device_capability()

    model = SE3Transformer(
        num_layers=args.num_layers,
        fiber_in=Fiber({0: args.node_dim}),
        fiber_hidden=Fiber.create(args.num_degrees, args.num_channels),
        fiber_out=Fiber({0: args.num_degrees * args.num_channels}),
        fiber_edge=Fiber({0: args.edge_dim}),
        num_heads=args.num_heads,
        channels_div=args.channels_div,
        return_type=0,
        pooling=None,
        norm=True,
        use_layer_norm=True,
        tensor_cores=(use_amp and major_cc >= 7) or major_cc >= 8,
        low_memory=False,
    ).to(device)
    model.eval()

    graph, node_feats, edge_feats = build_batched_inputs(
        batch_size=args.batch_size,
        num_nodes=args.num_nodes,
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        device=device,
    )

    if cuda_available:
        torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(args.warmup_iters):
            with torch.cuda.amp.autocast(enabled=use_amp):
                _ = model(graph, node_feats, edge_feats)
        if cuda_available:
            torch.cuda.synchronize()

        activities = [torch.profiler.ProfilerActivity.CPU]
        if cuda_available:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(activities=activities) as profiler:
            for _ in range(args.profile_iters):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    _ = model(graph, node_feats, edge_feats)
            if cuda_available:
                torch.cuda.synchronize()

    profiler.export_chrome_trace(str(args.output_dir / "trace.json"))
    key_averages = profiler.key_averages()
    write_table(key_averages, args.output_dir, "top_cuda_total.txt", ["cuda_time_total", "cpu_time_total"])
    write_table(key_averages, args.output_dir, "top_cuda_self.txt", ["self_cuda_time_total", "self_cpu_time_total"])
    write_table(key_averages, args.output_dir, "top_memory.txt", ["self_cuda_memory_usage", "cuda_memory_usage", "self_cpu_memory_usage"])

    print("Wrote profiler outputs:")
    print(f"  - {args.output_dir / 'trace.json'}")
    print(f"  - {args.output_dir / 'top_cuda_total.txt'}")
    print(f"  - {args.output_dir / 'top_cuda_self.txt'}")
    print(f"  - {args.output_dir / 'top_memory.txt'}")


if __name__ == "__main__":
    main()
