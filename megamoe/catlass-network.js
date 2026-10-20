/* 非抢占整块独占以外的通信模型：同方向、同资源的活跃请求共享带宽。 */
(function(root){
  "use strict";
  function schedule(nodes,c,initial,prepare,MinHeap){
    const future=new MinHeap((a,b)=>a.time-b.time||a.id-b.id);
    const flows=new Map(),usage=[];
    let now=initial,done=0;
    const addEvent=(time,id,type)=>future.push({time,id,type});
    function release(n){
      n.release=n.start=now;
      if(n.commKind){
        Object.assign(n,prepare(n));
        n.pendingPaths=n.paths.length;
        addEvent(now+(n.issueUs||0),n.id,"issue");
      }else addEvent(now+n.duration,n.id,"done");
    }
    function finish(n){
      n.end=now;done++;
      for(const id of n.next){const next=nodes[id];if(--next.pending===0)release(next);}
    }
    nodes.forEach(n=>{n.pending=n.deps.length;n.deps.forEach(id=>nodes[id].next.push(n.id));});
    nodes.filter(n=>!n.pending).forEach(release);
    while(done<nodes.length){
      // 先处理同一时刻的全部就绪任务，再计算共享份额，避免 core/id 顺序抢占整条链路。
      while(future.length && future.peek().time<=now+1e-9){
        const event=future.pop(),n=nodes[event.id];
        if(event.type==="done"){finish(n);continue;}
        if(!n.paths.length){finish(n);continue;}
        for(const p of n.paths){
          p.start=now;p.remaining=p.serviceUs;
          const key=n.commKind+"/"+p.path;
          if(!flows.has(key))flows.set(key,[]);
          flows.get(key).push({n,p});
        }
      }
      if(done===nodes.length)break;
      let next=future.length?future.peek().time:Infinity;
      for(const group of flows.values())if(group.length){
        const divisor=c.catlassNetwork==="fixed"&&group[0].p.path==="Clos"?c.cores:group.length;
        for(const f of group)next=Math.min(next,now+f.p.remaining*divisor);
      }
      if(!Number.isFinite(next))throw new Error(`CATLASS：依赖死锁 ${done}/${nodes.length}`);
      const dt=Math.max(0,next-now),completed=[];
      for(const [resource,group] of flows){
        if(!group.length)continue;
        const divisor=c.catlassNetwork==="fixed"&&group[0].p.path==="Clos"?c.cores:group.length;
        usage.push({resource,start:now,end:next,active:group.length,capacityGbps:group[0].p.bandwidth,
          usedGbps:group[0].p.bandwidth*group.length/divisor});
        for(const f of group){
          f.p.remaining-=dt/divisor;
          if(f.p.remaining<=1e-8){f.p.end=next;delete f.p.remaining;completed.push(f);}
        }
        flows.set(resource,group.filter(f=>f.p.remaining!==undefined));
      }
      now=next;
      for(const {n} of completed)if(--n.pendingPaths===0)addEvent(now,n.id,"done");
    }
    return usage;
  }
  const api={schedule};
  if(typeof module!=="undefined"&&module.exports)module.exports=api;else root.CatlassNetwork=api;
})(globalThis);
