"""CPU 数学验证：证明用例和反例，不启动 NPU，不代表真实路由或性能验收。"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261113)


def score(x):
    return np.sqrt(np.logaddexp(0.0, x))


def topk(x, k):
    # 分数相同按专家 ID，证书使用严格间隔，避开并列边界。
    return np.lexsort((np.arange(x.size), -x))[:k]


def main():
    cases = {}
    m, r, d, j, h, e = 5, 4, 7, 9, 11, 6
    A, W1, W2, Wg = [RNG.normal(size=s) for s in ((m,d),(d,j),(j,h),(h,e))]
    R = RNG.normal(size=(m,r,h))
    C = RNG.normal(size=(m,r,r))
    b, a = RNG.normal(size=(m,r)), RNG.normal(size=(m,r))
    gamma = RNG.normal(size=h)
    y = A @ W1 @ W2
    R0 = np.einsum('mij,mjh->mih', C, R) + b[:,:,None]*y[:,None,:]
    x = np.einsum('mi,mih->mh', a, R0)
    beta = np.einsum('mi,mij->mj', a, C)
    alpha = np.einsum('mi,mi->m', a, b)
    residual = np.einsum('mi,mih->mh', beta, R)
    folded_x = residual + alpha[:,None]*y
    Q = gamma[:,None]*Wg
    q = x @ Q
    early_q = residual @ Q + alpha[:,None]*(A @ (W1 @ (W2 @ Q)))
    np.testing.assert_allclose(x, folded_x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q, early_q, rtol=1e-12, atol=1e-12)
    cases['shifted_mhc_expansion'] = dict(passed=True, max_abs_error=float(abs(q-early_q).max()), dtype='float64', shapes=dict(M=m,r=r,D=d,J=j,H=h,E=e))

    rho = np.sqrt(np.mean(x*x,axis=1)+1e-6)
    for row, scale in zip(q, rho):
        assert np.array_equal(topk(row,3),topk(score(row/scale),3))
    cases['no_bias_positive_scale_and_monotone_activation'] = dict(passed=True)

    raw=np.array([2.,1.]); bias=np.array([0.,.1])
    before=score(raw)+bias; after=score(raw*.1)+bias
    assert topk(before,1)[0] != topk(after,1)[0]
    cases['scale_with_post_activation_bias_counterexample'] = dict(passed=True, raw=raw.tolist(), bias=bias.tolist(), scores_scale1=before.tolist(), scores_scale01=after.tolist())
    w0=score(raw)/score(raw).sum(); w1=score(raw*.1)/score(raw*.1).sum()
    assert not np.allclose(w0,w1)
    cases['same_ids_do_not_imply_same_weights'] = dict(passed=True, weights_scale1=w0.tolist(), weights_scale01=w1.tolist())

    # gamma 和旧残差即使只用两个通道，也可改变专家 ID。
    assert topk(raw,1)[0] != topk(raw*np.array([.1,10.]),1)[0]
    assert topk(raw,1)[0] != topk(raw+np.array([0.,3.]),1)[0]
    cases['gamma_and_residual_cannot_be_omitted'] = dict(passed=True)

    u=A@W1; G=W2@W2.T
    n2=np.sum(residual*residual,axis=1)+2*alpha*np.sum(u*(residual@W2.T),axis=1)+alpha**2*np.einsum('mi,ij,mj->m',u,G,u)
    direct_n2=np.sum(folded_x*folded_x,axis=1)
    np.testing.assert_allclose(n2,direct_n2,rtol=1e-12,atol=1e-12)
    cases['gram_norm_identity'] = dict(passed=True,max_abs_error=float(abs(n2-direct_n2).max()))

    # 这里的 C_F,b_F 是本次 MoE 的写回系数，与前面的 C_A,b_A 区分。
    Cf=RNG.normal(size=(m,r,r)); bf=RNG.normal(size=(m,r))
    contributions=RNG.normal(size=(m,7,h))
    v=np.sum(contributions,axis=1)
    post=np.einsum('mij,mjh->mih',Cf,R0)+bf[:,:,None]*v[:,None,:]
    distributed=np.einsum('mij,mjh->mih',Cf,R0)+np.sum(bf[:,None,:,None]*contributions[:,:,None,:],axis=1)
    np.testing.assert_allclose(post,distributed,rtol=1e-12,atol=1e-12)
    cases['tail_affine_distribution'] = dict(passed=True,max_abs_error=float(abs(post-distributed).max()),residual_added_once=True)

    entries=RNG.normal(size=(m,24,4));wk=RNG.normal(size=(24,4,13))
    direct_kv=entries.reshape(m,-1)@wk.reshape(24*4,13)
    grouped_kv=sum(entries[:,i,:]@wk[i] for i in range(24))
    np.testing.assert_allclose(direct_kv,grouped_kv,rtol=1e-12,atol=1e-12)
    cases['engram_concat_projection_distribution'] = dict(passed=True,max_abs_error=float(abs(direct_kv-grouped_kv).max()),dtype='float64',scope='toy blocks; not real quantized projection')

    # 有界尺度的区间证书：证明允许发 metadata 的条件，不把猜测当精确 TopK。
    certified=0
    for _ in range(200):
        raw=RNG.normal(size=16); bias=RNG.normal(scale=.02,size=16)
        lo,hi=.8,1.2
        left,right=score(raw/lo)+bias,score(raw/hi)+bias
        lower,upper=np.minimum(left,right),np.maximum(left,right)
        ids=topk(score(raw)+bias,3); rest=np.setdiff1d(np.arange(16),ids)
        if np.min(lower[ids]) > np.max(upper[rest]):
            certified+=1
            for norm in np.linspace(lo,hi,101):
                assert set(ids)==set(topk(score(raw/norm)+bias,3))
    assert 0<certified<200
    cases['interval_certificate_toy'] = dict(passed=True,certified_cases=certified,total_cases=200,note='synthetic interval only; not a real-model hit rate; no finite-precision bound certified')

    z=np.array([1e20,-1e20,3.14],dtype=np.float32)
    first=float(np.float32(np.float32(z[0]+z[1])+z[2])); second=float(np.float32(z[0]+np.float32(z[1]+z[2])))
    assert first != second
    cases['real_equivalence_not_bitwise_equivalence'] = dict(passed=True,left_associated=first,right_associated=second)
    result=dict(scope='CPU float64 identities and explicit counterexamples only; no NPU experiment or speed claim',seed=20261113,checks=cases,
        shape_costs=dict(D=32768,J=8192,H=5120,E=384,full_early_matrix_elements=32768*384,full_early_bf16_mib=32768*384*2/2**20,
            after_first_proj_bf16_mib=8192*384*2/2**20,gram_fp32_mib=8192**2*4/2**20,
            original_gate_macs_per_token=5120*384,full_attention_early_macs_per_token=32768*384))
    (ROOT/'math-checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,checks=len(cases),scope=result['scope']),ensure_ascii=True))


if __name__=='__main__': main()
