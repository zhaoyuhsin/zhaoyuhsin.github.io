
/* 单卡连续Base服务模型，位数与字节换算集中在此；不使用Tile活跃率代替吞吐效率。 */
(function(root){
  'use strict';
  const defaults={model:'ds-v4-pro',ep:64,gmmBits:8,peak8:865,computeUtil:100,dispatchBits:4,combineBits:16,networkUtil:100,clos:140,fm:34.81,group:8,scaleGroup:32,scaleBytes:1,includeShared:false};
  const models=[
    {id:'ds-v4-pro',name:'DeepSeek V4 Pro',I:3072,experts:384,K:6,S:1},
    {id:'ds-v41-flash',name:'DeepSeek V4.1 flash',I:2304,experts:384,K:6,S:1},
    {id:'ds-next10',name:'DeepSeek Next-10T',I:3072,experts:2048,K:18,S:2},
    {id:'ds-next20',name:'DeepSeek Next-20T',I:4096,experts:3072,K:20,S:4},
    {id:'hy-preview',name:'hunyuan 4 preview',I:2048,experts:256,K:8,S:1},
    {id:'hy-next10',name:'hunyuan Next-10T',I:3072,experts:2048,K:18,S:2},
    {id:'kimi-k3',name:'Kimi K3',I:3072,experts:896,K:16,S:2},
    {id:'kimi-next10',name:'Kimi Next-10T',I:4096,experts:2048,K:32,S:2},
    {id:'kimi-next20',name:'Kimi Next-20T',I:4096,experts:3072,K:48,S:2},
    {id:'glm-52',name:'GLM 5.2/5.3',I:2048,experts:256,K:8,S:1},
    {id:'glm-next10',name:'GLM Next-10T',I:3072,experts:2048,K:18,S:2},
    {id:'qwen-max',name:'Qwen 3.8-Max-2.4T-A95B',I:2048,experts:512,K:10,S:1},
    {id:'qwen-35',name:'Qwen 3.5-35B-A3B',I:512,experts:256,K:8,S:1}
  ];
  function validate(p){
    for(const k of ['ep','group','scaleGroup'])if(!Number.isSafeInteger(p[k])||p[k]<(k==='group'?2:1))throw Error(k+' 必须为范围内正整数（FullMesh组≥2卡）');
    if(![8,16,32,64,128,256].includes(p.ep)||p.group>4096||p.scaleGroup>1048576)throw Error('EP请选择8/16/32/64/128/256，FullMesh组和Scale组大小应在范围内');
    if(p.ep>p.group&&p.ep%p.group)throw Error('FullMesh每组卡数须整除EP，或覆盖整个EP组');
    for(const k of ['peak8','clos','fm'])if(!(Number.isFinite(p[k])&&p[k]>0))throw Error(k+' 必须为有限正数');
    for(const k of ['computeUtil','networkUtil'])if(!(Number.isFinite(p[k])&&p[k]>0&&p[k]<=100))throw Error('算力/通信利用率必须大于0且不超过100%');
    if(![4,8].includes(p.gmmBits)||![4,8].includes(p.dispatchBits)||![8,16].includes(p.combineBits))throw Error('不支持的GMM/通信精度');
    if(!(Number.isFinite(p.scaleBytes)&&p.scaleBytes>=0))throw Error('Scale字节必须为有限非负数');
    if(typeof p.includeShared!=='boolean')throw Error('共享专家开关必须为布尔值');
    if(!models.some(m=>m.id===p.model))throw Error('未知观察模型');
  }
  function calculate(model,options={}){
    const p={...defaults,...options};validate(p);
    if(!model||!Number.isSafeInteger(model.I)||model.I<=0||!Number.isSafeInteger(model.experts)||model.experts<=0||!Number.isSafeInteger(model.K)||model.K<=0||model.K>model.experts||!Number.isSafeInteger(model.S)||model.S<0)throw Error('模型I/专家数/TopK/共享专家参数无效');
    if(model.experts%p.ep)return {valid:false,model,error:`${model.experts}专家不能均分到EP${p.ep}`};
    const closPeers=p.ep-Math.min(p.ep,p.group),fmCost=1/p.fm,closCost=closPeers/p.clos;
    const bottleneckCost=Math.max(fmCost,closCost),weff=1/bottleneckCost;
    const peak=p.peak8*(8/p.gmmBits),etaC=p.computeUtil/100,etaN=p.networkUtil/100;
    const dispatchBytes=p.dispatchBits/8+p.scaleBytes/p.scaleGroup;
    const combineBytes=p.combineBits/8+(p.combineBits===8?p.scaleBytes/p.scaleGroup:0);
    const sharedFactor=p.includeShared?1+model.S/model.K:1;
    // 单位bs×K×H的服务秒数；相除时bs、K和H无需虚构具体取值。
    const computeSeconds=model.I*6*sharedFactor/(peak*1e12*etaC);
    const commSeconds=(dispatchBytes+combineBytes)*bottleneckCost/(p.ep*1e9*etaN);
    const ratio=computeSeconds/commSeconds;
    if(!(Number.isFinite(ratio)&&ratio>0))throw Error('参数量级导致比例溢出或下溢，请缩小数值范围');
    // 两条路径都要达到计算时长约束；不能在FullMesh限速时直接用当前Clos/ρ。
    const requiredFm=weff/ratio,requiredClos=closPeers*requiredFm;
    if(!Number.isFinite(requiredFm)||!Number.isFinite(requiredClos))throw Error('带宽阈值超出可表示范围，请调整参数量级');
    const fmEnoughForBalance=p.fm>=requiredFm*(1-1e-12);
    const balanceBandwidth=closPeers>0?requiredClos:requiredFm,balanceBandwidthKind=closPeers>0?'Clos单卡':'FM每Peer';
    return {valid:true,model,ratio,computeSeconds,commSeconds,closPeers,weff,effectiveWeff:weff*etaN,peak,effectivePeak:peak*etaC,dispatchBytes,combineBytes,sharedFactor,requiredFm,requiredClos,fmEnoughForBalance,balanceBandwidth,balanceBandwidthKind,expertsPerRank:model.experts/p.ep,path:closCost>fmCost?'Clos共享出口':'FullMesh专线',longer:Math.abs(ratio-1)<1e-9?'两者相等':ratio>1?'计算服务更长':'通信服务更长'};
  }
  root.MegaMoeRatio={defaults,models,validate,calculate};
})(globalThis);
