import time, random, sys
from concurrent.futures import ThreadPoolExecutor
from common import *
c=LLMCache('pilot2')
for task in sys.argv[1:]:
    us=load_proc(task)['users'][:10]; t=time.time()
    with ThreadPoolExecutor(4) as ex:
        B=list(ex.map(lambda u: audit_batch(c, AUDITOR, task, u['cands'][:5]), us))
        S=list(ex.map(lambda r: audit_record(c, AUDITOR, task, r), [r for u in us for r in u['cands'][:5]]))
    Bf=[x for b in B for x in b]; recs=[r for u in us for r in u['cands'][:5]]
    n=len(recs)
    print(task, n, 'agree(any)', sum(bool(a)==bool(b) for a,b in zip(Bf,S))/n, 'rate single', sum(map(bool,S))/n, 'rate batch', sum(map(bool,Bf))/n, 'sec', time.time()-t, flush=True)
    for r,a,b in list(zip(recs,Bf,S))[:25]: print('B',a,'S',b,'|',record_text(task,r)[:200])
