'use strict';
(() => {
  const api=globalThis.MegaMoeRatio;
  const output=document.getElementById('home-relation');
  if(!api){output.textContent='小实验暂时无法载入，请使用完整工具。';return;}
  const model=api.models.find(m=>m.id==='ds-v4-pro');
  let bits=8;
  const topology=document.getElementById('home-topology');
  const buttons=[...document.querySelectorAll('[data-bits]')];
  function update(){
    try{
      const result=api.calculate(model,{...api.defaults,gmmBits:bits,clos:Number(topology.value)});
      document.getElementById('home-ratio').textContent=result.ratio.toFixed(3);
      output.textContent=result.ratio>=1?'计算服务时间更长':'通信服务时间更长';
      const length=Math.max(result.ratio,1);
      document.getElementById('home-compute-bar').style.width=`${result.ratio/length*100}%`;
      document.getElementById('home-comm-bar').style.width=`${1/length*100}%`;
      document.getElementById('home-compute-label').textContent=result.ratio.toFixed(2);
      document.getElementById('home-target').textContent=`${result.requiredClos.toFixed(1)} GB/s`;
      buttons.forEach(button=>button.setAttribute('aria-pressed',String(Number(button.dataset.bits)===bits)));
    }catch(error){output.textContent='参数计算失败：'+error.message;}
  }
  buttons.forEach(button=>button.addEventListener('click',()=>{bits=Number(button.dataset.bits);update();}));
  topology.addEventListener('change',update);
  update();
})();
