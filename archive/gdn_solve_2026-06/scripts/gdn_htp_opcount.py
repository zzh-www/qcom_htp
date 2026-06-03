#!/usr/bin/env python3
"""Count lowered HTP ops from a ctxgen --save_backend_op_mapping bottom_mapping.json.
x86-only proxy for device op-count (the dispatch-overhead driver). Misses runtime SyncOps."""
import json,collections,re,sys
def count(path):
    d=json.load(open(path))
    # locate the htp_graph node dict
    def find_nodes(o):
        if isinstance(o,dict):
            if "nodes" in o and isinstance(o["nodes"],dict): yield o["nodes"]
            for v in o.values(): yield from find_nodes(v)
        elif isinstance(o,list):
            for v in o: yield from find_nodes(v)
    best={}
    for nd in find_nodes(d):
        if len(nd)>len(best): best=nd
    by_type=collections.Counter()
    by_src=collections.defaultdict(collections.Counter)
    for nid,n in best.items():
        t=n.get("type","?"); g=n.get("grouping","?")
        by_type[t]+=1
        src=re.sub(r'_\d+$','',str(g))
        by_src[src][t]+=1
    return len(best),by_type,by_src
if __name__=="__main__":
    n,bt,bs=count(sys.argv[1])
    print(f"total HTP nodes (ex-SyncOp): {n}")
    for t,c in bt.most_common(20): print(f"  {c:>4}  {t}")
    if len(sys.argv)>2:
        print("--- by source grouping ---")
        for src,cc in sorted(bs.items(),key=lambda x:-sum(x[1].values()))[:12]:
            print(f"  {sum(cc.values()):>4}  {src}: {dict(cc)}")
