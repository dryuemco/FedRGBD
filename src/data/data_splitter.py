"""FedRGBD Data Splitter — IID and Non-IID partitioning for 2 or 3 FL nodes."""

import argparse, json, os, random
from pathlib import Path

def find_images(data_dir):
    fire, nofire = [], []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                full = os.path.join(root, f)
                parent = os.path.basename(root).lower()
                if parent == 'fire':
                    fire.append(full)
                elif parent in ('no_fire', 'nofire'):
                    nofire.append(full)
    return fire, nofire

def split_list(items, train=0.7, val=0.15, seed=42):
    random.seed(seed)
    s = items.copy()
    random.shuffle(s)
    n = len(s)
    t, v = int(n*train), int(n*(train+val))
    return {'train': s[:t], 'val': s[t:v], 'test': s[v:]}

def link_files(file_list, dest_dir, class_name):
    dest = Path(dest_dir) / class_name
    dest.mkdir(parents=True, exist_ok=True)
    for src in file_list:
        link = dest / os.path.basename(src)
        if link.exists(): link.unlink()
        os.symlink(os.path.abspath(src), str(link))

def create_split(output_dir, split_name, node_data, seed):
    stats = {}
    for node_name, nd in node_data.items():
        fs = split_list(nd['fire'], seed=seed)
        ns = split_list(nd['nofire'], seed=seed)
        node_stats = {}
        for sp in ['train', 'val', 'test']:
            d = os.path.join(output_dir, split_name, node_name, sp)
            link_files(fs[sp], d, 'Fire')
            link_files(ns[sp], d, 'No_Fire')
            nf, nn = len(fs[sp]), len(ns[sp])
            node_stats[sp] = {'fire': nf, 'nofire': nn, 'total': nf+nn,
                             'fire_ratio': round(nf/max(nf+nn,1), 3)}
        stats[node_name] = node_stats
        total = sum(s['total'] for s in node_stats.values())
        tf = sum(s['fire'] for s in node_stats.values())
        print(f"  {node_name}: {total} imgs (Fire:{tf}, NoFire:{total-tf}, ratio:{tf/total:.1%})")
    return stats

def split_into(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw/flame_dataset")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nodes", type=int, default=3, choices=[2,3])
    args = parser.parse_args()

    print("=" * 50)
    print(f"  FedRGBD Data Splitter ({args.nodes} nodes)")
    print("=" * 50)

    fire, nofire = find_images(args.data_dir)
    print(f"\nFire: {len(fire)}, NoFire: {len(nofire)}, Total: {len(fire)+len(nofire)}")

    if not fire or not nofire:
        print("ERROR: No images found!"); return

    random.seed(args.seed)
    all_stats = {}
    node_names = ['node_a', 'node_b', 'node_c'][:args.nodes]

    # IID: equal random split
    print(f"\n--- IID Split ({args.nodes} nodes) ---")
    rf, rn = fire.copy(), nofire.copy()
    random.shuffle(rf); random.shuffle(rn)
    fire_parts = split_into(rf, args.nodes)
    nofire_parts = split_into(rn, args.nodes)
    iid_nodes = {name: {'fire': fp, 'nofire': np} 
                 for name, fp, np in zip(node_names, fire_parts, nofire_parts)}
    all_stats['iid'] = create_split(args.output_dir, 'iid', iid_nodes, args.seed)

    # Non-IID label skew
    print(f"\n--- Non-IID Label Skew ({args.nodes} nodes) ---")
    random.shuffle(rf); random.shuffle(rn)
    total = len(fire) + len(nofire)
    per_node = total // args.nodes

    if args.nodes == 2:
        a_fire = min(int(per_node*0.7), len(fire))
        a_nofire = per_node - a_fire
        noniid_nodes = {
            'node_a': {'fire': rf[:a_fire], 'nofire': rn[:a_nofire]},
            'node_b': {'fire': rf[a_fire:], 'nofire': rn[a_nofire:]}
        }
    else:
        # 3 nodes: A=80% fire, B=50/50, C=80% nofire
        a_fire = min(int(per_node*0.8), len(fire))
        a_nofire = per_node - a_fire
        c_nofire = min(int(per_node*0.8), len(nofire) - a_nofire)
        c_fire = per_node - c_nofire
        b_fire = len(fire) - a_fire - c_fire
        b_nofire = len(nofire) - a_nofire - c_nofire
        noniid_nodes = {
            'node_a': {'fire': rf[:a_fire], 'nofire': rn[:a_nofire]},
            'node_b': {'fire': rf[a_fire:a_fire+b_fire], 'nofire': rn[a_nofire:a_nofire+b_nofire]},
            'node_c': {'fire': rf[a_fire+b_fire:], 'nofire': rn[a_nofire+b_nofire:]}
        }

    all_stats['non_iid_label'] = create_split(args.output_dir, 'non_iid_label', noniid_nodes, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'split_stats.json'), 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved to {args.output_dir}/split_stats.json")

if __name__ == "__main__":
    main()
