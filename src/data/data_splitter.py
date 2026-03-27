"""FedRGBD Data Splitter — IID and Non-IID partitioning for 2 FL nodes."""

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

def create_split(fire, nofire, output_dir, split_name, node_data, seed):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw/flame_dataset")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 50)
    print("  FedRGBD Data Splitter")
    print("=" * 50)

    fire, nofire = find_images(args.data_dir)
    print(f"\nFire: {len(fire)}, NoFire: {len(nofire)}, Total: {len(fire)+len(nofire)}")

    if not fire or not nofire:
        print("ERROR: No images found!"); return

    random.seed(args.seed)
    all_stats = {}

    # IID: random 50/50
    print("\n--- IID Split ---")
    rf, rn = fire.copy(), nofire.copy()
    random.shuffle(rf); random.shuffle(rn)
    mf, mn = len(rf)//2, len(rn)//2
    iid_nodes = {
        'node_a': {'fire': rf[:mf], 'nofire': rn[:mn]},
        'node_b': {'fire': rf[mf:], 'nofire': rn[mn:]}
    }
    all_stats['iid'] = create_split(fire, nofire, args.output_dir, 'iid', iid_nodes, args.seed)

    # Non-IID label skew: Node A 70% Fire, Node B 70% NoFire
    print("\n--- Non-IID Label Skew ---")
    random.shuffle(rf); random.shuffle(rn)
    half = (len(fire)+len(nofire))//2
    a_fire = min(int(half*0.7), len(fire))
    a_nofire = half - a_fire
    noniid_nodes = {
        'node_a': {'fire': rf[:a_fire], 'nofire': rn[:a_nofire]},
        'node_b': {'fire': rf[a_fire:], 'nofire': rn[a_nofire:]}
    }
    all_stats['non_iid_label'] = create_split(fire, nofire, args.output_dir, 'non_iid_label', noniid_nodes, args.seed)

    # Save stats
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'split_stats.json'), 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved to {args.output_dir}/split_stats.json")

if __name__ == "__main__":
    main()
