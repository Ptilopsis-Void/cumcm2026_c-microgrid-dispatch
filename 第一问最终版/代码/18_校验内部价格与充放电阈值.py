from pathlib import Path
import importlib.util
import hashlib
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT / "代码"
RESULT_DIR = PROJECT / "模型结果"

OUT_ACCEPT = RESULT_DIR / "第一问_对偶与KKT验收.csv"

J_LOCK = 35126.9485892896
SOL_FROZEN_SHA = "6c1bcd91155d71ca5ae0f09762b40a6d5b98781454c6730bbf8fcafda17ced0a"

EPS_PRICE = 1e-7
EPS_RC = 1e-7
EPS_COMP = 1e-6
EPS_VAR = 1e-7
EPS_BOUND = 1e-5

SOLUTION_CSV = RESULT_DIR / "最终LP_原始最优解.csv"


def _load17():
    spec = importlib.util.spec_from_file_location("m17", str(CODE_DIR / "17_提取第一问对偶价格.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def _rng(cond):
    return [int(i) for i in np.where(cond)[0] + 1]


def main():
    sha_before = sha256(RESULT_DIR / "最终LP_原始最优解.csv")
    m17 = _load17()
    d = m17.main()

    c = d["arr"]["c"]; g = d["arr"]["g"]; U = d["arr"]["U"]
    pi = d["pi"]; lam = d["lam"]
    C = d["arr"]["C"]; D = d["arr"]["D"]
    rc_c = d["arr"]["rc_c"]; rc_d = d["arr"]["rc_d"]
    S = m17.S

    C_upper = C >= S - EPS_BOUND
    C_zero = C <= EPS_VAR
    C_int = ~(C_upper | C_zero)
    D_upper = D >= S - EPS_BOUND
    D_zero = D <= EPS_VAR
    D_int = ~(D_upper | D_zero)

    j_real = float((c * g).sum())
    q_real = float(g.sum())
    sol_sha = sha256(SOLUTION_CSV)

    checks = []

    def _add(name, violation, tol, cond_intervals, explanation):
        ok = bool(violation <= tol)
        checks.append({
            "check_name": name,
            "max_violation": float(violation),
            "tolerance": float(tol),
            "pass": ok,
            "affected_intervals": ",".join(str(x) for x in cond_intervals) if cond_intervals else "",
            "explanation": explanation,
        })

    v = float(np.maximum(-pi, 0.0).max())
    _add("π_t ≥ 0", v, EPS_PRICE, _rng(pi < -EPS_PRICE),
         "内部价格非负（对应 U_t 下界约化成本 ∂L/∂U=π 非负）")
    v = float(np.maximum(pi - c, 0.0).max())
    _add("π_t ≤ c_t", v, EPS_PRICE, _rng(pi > c + EPS_PRICE),
         "内部价格不超过外网电价（对应 g_t 下界约化成本 ∂L/∂g=c-π 非负）")
    gc = g * (c - pi)
    v = float(np.max(np.abs(gc)))
    _add("g_t(c_t-π_t)=0", v, EPS_COMP, _rng(np.abs(gc) > EPS_COMP),
         "正购电时段 π=c，故 g(c-π)=0（元）")
    pu = pi * U
    v = float(np.max(np.abs(pu)))
    _add("π_t·U_t=0", v, EPS_COMP, _rng(np.abs(pu) > EPS_COMP),
         "有弃光时段 π=0，故 πU=0（元）")
    v = float((-rc_c[C_zero]).max()) if C_zero.any() else 0.0
    _add("充电下界 KKT（C=0 ⇒ π-ηλ≥0）", max(v, 0.0), EPS_RC,
         _rng(C_zero & (rc_c < -EPS_RC)),
         "零充电时充电约化成本非负")
    v = float(np.abs(rc_c[C_int]).max()) if C_int.any() else 0.0
    _add("充电内部点 KKT（0<C<S ⇒ π-ηλ=0）", v, EPS_RC,
         _rng(C_int & (np.abs(rc_c) > EPS_RC)),
         "内部充电约化成本为零")
    v = float(rc_c[C_upper].max()) if C_upper.any() else 0.0
    _add("充电上界 KKT（C=S ⇒ π-ηλ≤0）", max(v, 0.0), EPS_RC,
         _rng(C_upper & (rc_c > EPS_RC)),
         "满充时充电约化成本非正")
    v = float((-rc_d[D_zero]).max()) if D_zero.any() else 0.0
    _add("放电下界 KKT（D=0 ⇒ -π+λ/η≥0）", max(v, 0.0), EPS_RC,
         _rng(D_zero & (rc_d < -EPS_RC)),
         "零放电时放电约化成本非负")
    v = float(np.abs(rc_d[D_int]).max()) if D_int.any() else 0.0
    _add("放电内部点 KKT（0<D<S ⇒ -π+λ/η=0）", v, EPS_RC,
         _rng(D_int & (np.abs(rc_d) > EPS_RC)),
         "内部放电约化成本为零")
    v = float(rc_d[D_upper].max()) if D_upper.any() else 0.0
    _add("放电上界 KKT（D=S ⇒ -π+λ/η≤0）", max(v, 0.0), EPS_RC,
         _rng(D_upper & (rc_d > EPS_RC)),
         "满放时放电约化成本非正")
    v = float(abs(j_real - J_LOCK))
    _add("原始LP目标值未改变", v, 1e-6, [],
         f"当前费用 {j_real:.10f} 元 vs 锁定 {J_LOCK} 元")
    ok12 = sha256(RESULT_DIR / "最终LP_原始最优解.csv") == sha_before
    checks.append({
        "check_name": "原始LP解文件未改变",
        "max_violation": 0.0,
        "tolerance": 0.0,
        "pass": ok12,
        "affected_intervals": "",
        "explanation": f"SHA-256={sol_sha}" + ("（校验前后文件一致）" if ok12 else "（不一致！）"),
    })

    acc_df = pd.DataFrame(checks)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    acc_df.to_csv(OUT_ACCEPT, index=False, encoding="utf-8-sig")
    n_pass = int(acc_df["pass"].sum())

    print(f"[18] KKT 验收：{n_pass}/12 项通过")
    print(f"[18] 目标值差={abs(j_real-J_LOCK):.3e} 元；购电量={q_real:.10f} kWh")
    print(f"[18] 已导出 {OUT_ACCEPT.name}")
    return {
        "checks": checks, "acc_df": acc_df, "n_pass": n_pass,
        "j_real": j_real, "q_real": q_real, "sol_sha": sol_sha,
    }


if __name__ == "__main__":
    main()
