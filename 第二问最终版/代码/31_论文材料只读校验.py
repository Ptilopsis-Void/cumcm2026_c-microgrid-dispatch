from pathlib import Path
import csv, hashlib, json, importlib.util, sys, re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def read_csv(name):
    with (ROOT/name).open(encoding='utf-8-sig') as f:return list(csv.reader(f))

def md_table(rows):
    return '\n'.join(['| '+' | '.join(rows[0])+' |','|'+'|'.join(['---']*len(rows[0]))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows[1:]])+'\n'

def run_checks(emit=print):
    results=[]
    def check(name,ok):
        results.append((name,bool(ok)));emit(('通过 ' if ok else '失败 ')+name)
    protected_path=ROOT/'历史模型/论文口径修订前/主结果保护清单.json'
    if not protected_path.is_file():
        emit('跳过：未提供开发期论文防漂移保护清单；本脚本不属于正式复现必经链。')
        return [('开发期论文防漂移检查已跳过',True)]
    protected=json.loads(protected_path.read_text())
    check('本轮保护清单：正式Excel、模型结果、处理后数据、配置及图片未变',all((ROOT/f).is_file() and hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in protected.items()))
    z=np.load(ROOT/'处理后数据/附件二_矩阵数据.npz');bt=np.load(ROOT/'模型结果/第二问_执行器回测.npz'); ix=z['score_day_index'];dates=list(z['dates']);price=z['price']
    g=bt['plan_g'];b=bt['dp_b'];bill=float(((g+5*b)[ix]*price).sum())
    core=dict((r[0],float(r[1])) for r in read_csv('模型结果/第二问_最终核心结果.csv')[1:])
    check('年度账单从轨迹独立复算',abs(bill-core['全年总费用_元'])<1e-6)
    readme=(ROOT/'README.md').read_text();main=readme.split('<!-- MAIN_RESULTS_BEGIN -->')[1].split('<!-- MAIN_RESULTS_END -->')[0]
    names=['第二问_指定时段计划购电量.csv','第二问_指定日期全天购电量与费用.csv','第二问_指定日期费用分解.csv','第二问_指定日期紧急购电事件.csv','第二问_指定日期储能充放电与储电量.csv','第二问_最终核心结果.csv']
    check('README全部主结果表与当前CSV完整一致',all(md_table(read_csv('模型结果/'+n)) in main for n in names))
    normalized=re.sub(r'(?<=\d)[,，\s](?=\d)','',main)
    check('主结果区域无旧外部费用与总量',not any(x in normalized for x in ['14022279.33','20808141.202','204387.312','21012528.514','12245046.92']))
    a=pd.read_csv(ROOT/'论文补充材料/表A_指定日期购电量.csv');ok=True
    for _,r in a.iterrows():
        day=dates.index(r['日期']);key=r['项目']
        if '-' in key:
            h,m=map(int,key.split('-')[0].split(':'));expected=g[day,h*6+m//10]
        elif key=='六个指定时段计划量合计':expected=g[day,[60,72,84,96,108,120]].sum()
        elif key=='全天计划购电量':expected=g[day].sum()
        elif key=='全天紧急购电量':expected=b[day].sum()
        elif key=='全天购电费':expected=float((price*(g[day]+5*b[day])).sum())
        else:ok=False;continue
        ok &= abs(float(r['数值'])-expected)<1e-6
    check('表A各数值与主轨迹逐项一致',ok)
    bb=pd.read_csv(ROOT/'论文补充材料/表B_指定日期充放电.csv');ok=True
    for _,r in bb.iterrows():
        day=dates.index(r['日期']);key=r['时间段']
        if '-' in key:
            hour=int(key.split(':')[0]);lo=hour*6
            ok &= abs(r['充电量_kWh']-bt['dp_C'][day,lo:lo+24].sum())<1e-6
            ok &= abs(r['放电量_kWh']-bt['dp_D'][day,lo:lo+24].sum())<1e-6
        else:
            field='0:00储电量_kWh' if key.startswith('0:00') else '24:00储电量_kWh'
            val=bt['dp_E'][day-1,-1] if key.startswith('0:00') else bt['dp_E'][day,-1]
            ok &= abs(r[field]-val)<1e-6
    check('表B六段与首末库存逐项一致',ok)
    ec=pd.read_csv(ROOT/'论文补充材料/表C_指定日期紧急购电.csv');expected=[]
    for date in a['日期'].unique():
        day=dates.index(date);mask=b[day]>1e-6
        starts=np.flatnonzero(mask & ~np.r_[False,mask[:-1]]);ends=np.flatnonzero(mask & ~np.r_[mask[1:],False])+1
        for lo,hi in zip(starts,ends):expected.append((date,f'{lo//6:02d}:{lo%6*10:02d}-{hi//6:02d}:{hi%6*10:02d}',b[day,lo:hi].sum()))
    got=[(r['日期'],r['紧急购电时间段'],r['购电量_kWh']) for _,r in ec.iterrows()]
    check('表C事件完整性',len(got)==len(expected) and all(x[:2]==y[:2] and abs(x[2]-y[2])<1e-6 for x,y in zip(got,expected)))
    spec=importlib.util.spec_from_file_location('paper30',ROOT/'代码/30_补充材料_表格生成.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    expected_text={};expected_csv={};m.write=lambda name,text:expected_text.update({name:text});m.p=lambda *args:None
    original=pd.DataFrame.to_csv
    def capture(frame,path,*args,**kwargs):expected_csv[Path(path).name]=original(frame,None,*args,**kwargs)
    pd.DataFrame.to_csv=capture
    try:
        d=m.load_all();m.table_a(d);m.table_b(d);m.table_c(d);ns,keep=m.table_d(d);cn=m.table_e(d,ns,keep);m.table_f(d,ns,keep);m.table_h(d)
        from _paper_guidance import build_guidance
        draft,guide=build_guidance(d,cn);expected_text.update({'段落草稿.md':draft,'README.md':guide})
    finally:pd.DataFrame.to_csv=original
    check('论文Markdown与已修正生成模板一致',all((ROOT/'论文补充材料'/n).read_text()==text for n,text in expected_text.items()))
    check('论文CSV与同源表格一致',all((ROOT/'论文补充材料'/n).read_text(encoding='utf-8-sig')==text for n,text in expected_csv.items()))
    spec=importlib.util.spec_from_file_location('read17',ROOT/'代码/17_最终定稿与交付.py');m17=importlib.util.module_from_spec(spec);spec.loader.exec_module(m17)
    _,fail=m17.readback_result2(ROOT/'提交结果/result2.xlsx',m17.load_data(),lambda *args:None)
    check('正式Excel只读回读',fail==0)
    emit(f'{sum(ok for _,ok in results)}/{len(results)} 项通过；未写入任何结果文件。')
    return results

if __name__=='__main__':
    raise SystemExit(int(not all(ok for _,ok in run_checks())))
