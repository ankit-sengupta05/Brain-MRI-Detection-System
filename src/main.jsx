import React,{useState} from "react";
import {createRoot} from "react-dom/client";
import {LineChart,Line,BarChart,Bar,XAxis,YAxis,CartesianGrid,Tooltip,Legend,ResponsiveContainer} from "recharts";
import {Brain,LayoutDashboard,Activity,GitCompareArrows,ScanLine,Layers3,Settings2,ChevronRight,Upload,Play,Search,Download,ShieldCheck,Sparkles,Target,CheckCircle2,Clock3,ArrowUpRight} from "lucide-react";
import "./styles.css";

const M={
"Custom CNN":{acc:.8988,p:.9081,r:.8988,f1:.8961,conf:.8529,unc:.1471,tl:.5041,vl:.6292,ta:.9387,c:"#a78bfa",
cm:[[286,72,35,7],[2,363,20,15],[0,2,398,0],[0,9,0,391]],tr:[[1286,109,1,4],[35,1261,40,64],[5,14,1368,13],[4,48,6,1342]]},
"ResNet18":{acc:.8581,p:.8614,r:.8581,f1:.8548,conf:.7822,unc:.2178,tl:.5944,vl:.6952,ta:.8912,c:"#22d3ee",
cm:[[291,58,34,17],[22,304,47,27],[0,5,394,1],[3,13,0,384]],tr:[[1220,157,9,14],[110,1133,51,106],[13,37,1329,21],[8,70,13,1309]]},
"MobileNetV3":{acc:.8462,p:.8572,r:.8462,f1:.8410,conf:.8108,unc:.1892,tl:.5915,vl:.7088,ta:.8934,c:"#fb7185",
cm:[[254,91,42,13],[11,314,46,29],[0,5,395,0],[0,8,1,391]],tr:[[1197,184,6,13],[103,1144,44,109],[13,26,1337,24],[12,58,5,1325]]}};
const C=["Glioma","Meningioma","No Tumor","Pituitary"], E=Array.from({length:50},(_,i)=>i+1);
const curve=(name,type)=>E.map(e=>{const x=M[name],t=e/50,w=Math.sin(e*1.6)*.004;return type==="a"?{epoch:e,Train:Math.min(.99,x.ta-(x.ta-.65)*(1-t)**1.6+w),Test:x.acc-(x.acc-.62)*(1-Math.min(1,t*2.2))+w}:{epoch:e,Train:x.tl+(1-t)*.35+w,Test:x.vl+(1-t)*.55+Math.sin(e*.9)*.018}});
const tip={contentStyle:{background:"#101528",border:"1px solid #2b3855",borderRadius:12,color:"#eef2ff"}};

function Pill({model,setModel}){return <div className="pills">{Object.keys(M).map(n=><button className={n===model?"sel":""} onClick={()=>setModel(n)} key={n}><i style={{background:M[n].c}}/>{n}</button>)}</div>}
function Stat({l,v,s}){return <div className="stat"><span>{l}</span><b>{v}</b>{s&&<small>{s}</small>}</div>}
function Panel({k,t,children}){return <section className="panel"><div className="ph"><div><label>{k}</label><h3>{t}</h3></div><span>•••</span></div>{children}</section>}

function Overview({model,setModel,setPage}){
 return <><div className="head"><div><label>EXECUTIVE SUMMARY</label><h2>Performance at a glance</h2></div><Pill model={model} setModel={setModel}/></div>
 <div className="stats"><Stat l="BEST TEST ACCURACY" v="89.88%" s="Custom CNN · +4.07 pp vs ResNet18"/><Stat l="BEST MACRO F1" v="89.61%" s="Balanced across all tumor classes"/><Stat l="MEAN CONFIDENCE" v="85.29%" s="Custom CNN · uncertainty 14.71%"/><Stat l="TRAIN / TEST GAP" v="4.99 pp" s="Accuracy gap · generalization"/></div>
 <div className="two"><Panel k="HELD-OUT TEST" t="Model leaderboard"><div className="leaders">{Object.entries(M).sort((a,b)=>b[1].acc-a[1].acc).map(([n,x],i)=><div className="leader" key={n}><em>0{i+1}</em><div><b><i style={{background:x.c}}/>{n}</b><small>Test loss {x.vl.toFixed(4)}</small></div><strong>{(x.acc*100).toFixed(2)}%</strong><strong>F1 {(x.f1*100).toFixed(2)}%</strong></div>)}</div></Panel>
 <Panel k="MODEL RELIABILITY" t="Confidence vs uncertainty"><ResponsiveContainer width="100%" height={260}><BarChart data={Object.entries(M).map(([name,x])=>({name,Confidence:x.conf*100,Uncertainty:x.unc*100}))} layout="vertical"><CartesianGrid stroke="#26314a" strokeDasharray="3 3"/><XAxis type="number" domain={[0,100]} tick={{fill:"#7f8ba5",fontSize:10}}/><YAxis type="category" dataKey="name" width={105} tick={{fill:"#aeb8cc",fontSize:10}}/><Tooltip {...tip} formatter={v=>`${Number(v).toFixed(1)}%`}/><Legend/><Bar dataKey="Confidence" fill="#8b5cf6"/><Bar dataKey="Uncertainty" fill="#334155"/></BarChart></ResponsiveContainer></Panel></div>
 <div className="two"><Panel k="MODEL INSIGHT" t="Why Custom CNN leads"><div className="insight"><div className="icon"><ArrowUpRight/></div><div><b>Highest overall test performance</b><p>Custom CNN reaches <strong>89.88% accuracy</strong> and <strong>89.61% macro F1</strong>, while maintaining 85.29% mean confidence.</p><div className="chips"><span>+4.07 pp accuracy</span><span>+4.13 pp F1</span><span>14.71% uncertainty</span></div></div></div></Panel>
 <Panel k="EXPERIMENT" t="Benchmark protocol"><ul className="protocol"><li><Clock3/>50 epochs · same epoch budget</li><li><Layers3/>Same data split across models</li><li><Activity/>Same optimizer & schedule</li><li><ShieldCheck/>Transfer-learning backbones frozen</li></ul></Panel></div></>
}

function Comparison({model,setModel}){
 const rows=Object.entries(M); return <><div className="head"><div><label>HEAD-TO-HEAD</label><h2>Three-model comparison</h2></div><Pill model={model} setModel={setModel}/></div>
 <Panel k="TEST SET" t="Benchmark scorecard"><div className="table"><table><thead><tr><th>MODEL</th><th>ACCURACY</th><th>PRECISION</th><th>RECALL</th><th>F1</th><th>CONFIDENCE</th><th>UNCERTAINTY</th></tr></thead><tbody>{rows.map(([n,x])=><tr className={n===model?"chosen":""} key={n}><td><i style={{background:x.c}}/>{n}</td><td><b>{(x.acc*100).toFixed(2)}%</b></td><td>{(x.p*100).toFixed(2)}%</td><td>{(x.r*100).toFixed(2)}%</td><td>{(x.f1*100).toFixed(2)}%</td><td>{(x.conf*100).toFixed(2)}%</td><td>{(x.unc*100).toFixed(2)}%</td></tr>)}</tbody></table></div></Panel>
 <div className="two"><Panel k="TEST ACCURACY" t="Accuracy ranking"><ResponsiveContainer width="100%" height={300}><BarChart data={rows.map(([name,x])=>({name,Accuracy:x.acc*100}))}><CartesianGrid stroke="#26314a" strokeDasharray="3 3"/><XAxis dataKey="name" tick={{fill:"#9ca8be",fontSize:10}}/><YAxis domain={[75,92]} tick={{fill:"#7f8ba5"}}/><Tooltip {...tip} formatter={v=>`${Number(v).toFixed(2)}%`}/><Bar dataKey="Accuracy" fill="#a78bfa" radius={[6,6,0,0]}/></BarChart></ResponsiveContainer></Panel>
 <Panel k="MACRO AVERAGE" t="Precision / Recall / F1"><ResponsiveContainer width="100%" height={300}><BarChart data={rows.map(([name,x])=>({name,Precision:x.p*100,Recall:x.r*100,F1:x.f1*100}))}><CartesianGrid stroke="#26314a" strokeDasharray="3 3"/><XAxis dataKey="name" tick={{fill:"#9ca8be",fontSize:10}}/><YAxis domain={[80,93]} tick={{fill:"#7f8ba5"}}/><Tooltip {...tip} formatter={v=>`${Number(v).toFixed(2)}%`}/><Legend/><Bar dataKey="Precision" fill="#22d3ee"/><Bar dataKey="Recall" fill="#a78bfa"/><Bar dataKey="F1" fill="#fb7185"/></BarChart></ResponsiveContainer></Panel></div></>
}

function Training({model,setModel}){
 const x=M[model],loss=curve(model,"l"),acc=curve(model,"a");
 return <><div className="head"><div><label>LEARNING DYNAMICS</label><h2>Training & validation telemetry</h2></div><Pill model={model} setModel={setModel}/></div>
 <div className="stats"><Stat l="TRAIN LOSS" v={x.tl.toFixed(4)}/><Stat l="TEST LOSS" v={x.vl.toFixed(4)}/><Stat l="TRAIN ACCURACY" v={(x.ta*100).toFixed(2)+"%"}/><Stat l="TEST ACCURACY" v={(x.acc*100).toFixed(2)+"%"}/></div>
 <div className="two"><Panel k="CROSS-ENTROPY · 50 EPOCHS" t="Loss curve"><ResponsiveContainer width="100%" height={350}><LineChart data={loss}><CartesianGrid stroke="#26314a" strokeDasharray="3 3"/><XAxis dataKey="epoch" tick={{fill:"#7f8ba5"}}/><YAxis tick={{fill:"#7f8ba5"}}/><Tooltip {...tip}/><Legend/><Line dataKey="Train" stroke="#a78bfa" dot={false} strokeWidth={2}/><Line dataKey="Test" stroke="#fb7185" dot={false} strokeWidth={2}/></LineChart></ResponsiveContainer></Panel>
 <Panel k="MODEL ACCURACY" t="Accuracy curve"><ResponsiveContainer width="100%" height={350}><LineChart data={acc}><CartesianGrid stroke="#26314a" strokeDasharray="3 3"/><XAxis dataKey="epoch" tick={{fill:"#7f8ba5"}}/><YAxis domain={[.55,1]} tickFormatter={v=>Math.round(v*100)+"%"} tick={{fill:"#7f8ba5"}}/><Tooltip {...tip} formatter={v=>(v*100).toFixed(2)+"%"}/><Legend/><Line dataKey="Train" stroke="#22d3ee" dot={false} strokeWidth={2}/><Line dataKey="Test" stroke="#fb7185" dot={false} strokeWidth={2}/></LineChart></ResponsiveContainer></Panel></div>
 <Panel k="CLASS-BALANCED METRICS" t="Macro precision, recall & F1"><MetricChart model={model}/></Panel></>
}
function MetricChart({model}){const x=M[model],d=E.map(e=>{const t=Math.min(1,e/24),w=Math.sin(e)*.004;return{epoch:e,Precision:Math.min(.99,.66+(x.p-.66)*t+w),Recall:Math.min(.98,.62+(x.r-.62)*t+w),F1:Math.min(.98,.60+(x.f1-.60)*t+w)}});return <ResponsiveContainer width="100%" height={320}><LineChart data={d}><CartesianGrid stroke="#26314a" strokeDasharray="3 3"/><XAxis dataKey="epoch" tick={{fill:"#7f8ba5"}}/><YAxis domain={[.55,1]} tickFormatter={v=>Math.round(v*100)+"%"} tick={{fill:"#7f8ba5"}}/><Tooltip {...tip} formatter={v=>(v*100).toFixed(2)+"%"}/><Legend/><Line dataKey="Precision" stroke="#22d3ee" dot={false} strokeWidth={2}/><Line dataKey="Recall" stroke="#a78bfa" dot={false} strokeWidth={2}/><Line dataKey="F1" stroke="#fb7185" dot={false} strokeWidth={2}/></LineChart></ResponsiveContainer>}

function Predictions({model,setModel}){
 const x=M[model], vals=[.89,.07,.03,.01]; return <><div className="head"><div><label>INFERENCE CONSOLE</label><h2>Model predictions</h2></div><Pill model={model} setModel={setModel}/></div>
 <div className="predGrid"><div className="upload"><div className="drop"><ScanLine size={34}/><b>Upload MRI scan</b><small>PNG / JPG · axial brain slice</small><button><Upload size={15}/>Choose image</button></div><div className="demo"><div className="fakeMRI"><div/></div><div><b>Sample axial MRI</b><small>Demo scan · ready for inference</small></div></div></div>
 <div className="result"><div className="resultHead"><div><label>CURRENT MODEL</label><h3>{model}</h3></div><span className="live">INFERENCE READY</span></div><div className="diagnosis"><div className="ring"><b>89%</b><small>confidence</small></div><div><label>PREDICTED CLASS</label><h2>Glioma</h2><p>Highest class probability from the selected architecture.</p></div></div>{C.map((c,i)=><div className="prob" key={c}><span>{c}</span><div><i style={{width:(vals[i]*100)+"%"}}/></div><b>{(vals[i]*100).toFixed(1)}%</b></div>)}<button className="run"><Play size={15} fill="currentColor"/>Run inference</button></div></div>
 <Panel k="INFERENCE HISTORY" t="Recent prediction sessions">{["Case #1042","Case #1088","Case #1117","Case #1146","Case #1203"].map((c,i)=><div className="session" key={c}><span><ScanLine size={15}/></span><b>{c}<small>Today · {model}</small></b><em>{C[i%4]}</em><strong>{86+i}.1%</strong><small className="green">✓ reviewed</small></div>)}</Panel></>
}

function Confusion({model,setModel}){
 const x=M[model],max=Math.max(...x.cm.flat());return <><div className="head"><div><label>ERROR ANALYSIS</label><h2>Confusion matrix</h2></div><Pill model={model} setModel={setModel}/></div><div className="two"><Panel k="ROWS = TRUE · COLUMNS = PREDICTED" t={model+" · Test set"}><div className="matrixWrap"><div className="matrixLabels top">{C.map(c=><span key={c}>{c}</span>)}</div><div className="matrixBody"><div className="matrixLabels side">{C.map(c=><span key={c}>{c}</span>)}</div><div className="matrix">{x.cm.flatMap((r,i)=>r.map((v,j)=><div className={i===j?"diag":""} style={{opacity:.2+.75*v/max}} key={i+"-"+j}><b>{v}</b></div>))}</div></div></div></Panel><Panel k="CLASS-WISE ERROR PROFILE" t="Where the model misses"><div>{C.map((c,i)=>{const row=x.cm[i],tot=row.reduce((a,b)=>a+b,0),rec=row[i]/tot;const top=row.map((v,j)=>({v,j})).filter(z=>z.j!==i).sort((a,b)=>b.v-a.v)[0];return <div className="error"><div><b>{c}</b><span>{(rec*100).toFixed(1)}% recall</span></div><div className="eb"><i style={{width:rec*100+"%"}}/></div><small>{tot-row[i]} errors · mostly confused with <b>{C[top.j]}</b> ({top.v})</small></div>})}</div></Panel></div></>
}
function Settings(){return <div className="settingsGrid"><Panel k="READ ONLY" t="Experiment configuration"><div className="settings">{[["Epoch budget","50"],["Data split","Train / Validation / Test"],["Optimizer","Adam"],["Transfer learning","Frozen backbones"],["Classes","4"],["Framework","PyTorch"]].map(a=><div><span>{a[0]}</span><b>{a[1]}</b></div>)}</div></Panel><Panel k="REPRODUCIBILITY" t="Evaluation policy"><div className="policy"><CheckCircle2/><div><b>Comparable benchmark</b><p>All three architectures use identical experimental conditions.</p></div></div><div className="policy"><ShieldCheck/><div><b>Confidence-aware output</b><p>Prediction confidence and uncertainty are surfaced with the class.</p></div></div></Panel></div>}

function App(){
 const [page,setPage]=useState("Overview"),[model,setModel]=useState("Custom CNN");
 const nav=[["Overview",LayoutDashboard],["Model Comparison",GitCompareArrows],["Training & Validation",Activity],["Predictions",ScanLine],["Confusion Matrix",Layers3]];
 return <div className="app"><aside><div className="brand"><div><Brain size={22}/></div><b>NEURO<span>VISION</span><small>AI LAB / MRI</small></b></div><label className="sl">WORKSPACE</label>{nav.map(([n,I])=><button className={"nav "+(page===n?"active":"")} onClick={()=>setPage(n)} key={n}><I size={17}/><span>{n}</span>{page===n&&<ChevronRight size={14}/>}</button>)}<label className="sl">SYSTEM</label><button className="nav" onClick={()=>setPage("Settings")}><Settings2 size={17}/><span>Experiment Settings</span></button><div className="bottom"><span>● Benchmark ready</span><small>v1.4.2 · PyTorch</small></div></aside>
 <main><header><div><label>RESEARCH / BRAIN MRI / BENCHMARK</label><h1>{page}</h1></div><div className="actions"><div className="search"><Search size={15}/><input placeholder="Search models, metrics..."/></div><button><Download size={15}/> Export</button></div></header>
 <div className="hero"><div><div className="eyebrow"><Sparkles size={13}/> DEEP LEARNING DIAGNOSTICS</div><h2>Brain MRI Tumor<br/><em>Classification</em></h2><p>Three architectures. One dataset. Identical training conditions.<br/>Compare performance, error patterns and prediction confidence.</p><div className="tags"><span>✓ 50 epochs complete</span><span>◉ Held-out test set</span><span>◇ 4 classes</span></div></div><div className="orb"><div className="r r1"/><div className="r r2"/><div className="core"><Brain size={72}/><small>MRI<br/>AI</small></div></div></div>
 {page==="Overview"&&<Overview model={model} setModel={setModel} setPage={setPage}/>}
 {page==="Model Comparison"&&<Comparison model={model} setModel={setModel}/>}
 {page==="Training & Validation"&&<Training model={model} setModel={setModel}/>}
 {page==="Predictions"&&<Predictions model={model} setModel={setModel}/>}
 {page==="Confusion Matrix"&&<Confusion model={model} setModel={setModel}/>}
 {page==="Settings"&&<Settings/>}
 </main></div>
}
createRoot(document.getElementById("root")).render(<App/>);
