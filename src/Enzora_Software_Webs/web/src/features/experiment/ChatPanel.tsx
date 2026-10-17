import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, isDemo } from "../../shared/api/client";
import { downloadText, serializeCsv } from "../../shared/lib/download";
import type { AtlasRecord,ProteinStructure,AssistantProgress } from "../../shared/api/contracts";
import { ProteinViewer } from "../../shared/ui/PlotWrap";
import { rememberRequest } from "../runs/requestStorage";
import { useApp } from "../../app/store";

type Message = {
  role: "agent" | "user";
  text: string;
  codons?: Record<string, unknown>[];
  structure?: ProteinStructure;
  atlas?: AtlasRecord;
  prepared?: {start:number;end:number;sequence:string};
  tools?: {tool:string}[];
  fasta?: {id:string;sequence:string}[];
};
export function ChatPanel({
  jobId,
  onChange,
  phase = 1,
  status,
  sequence,
}: {
  jobId: string | null;
  onChange: () => void;
  phase?: number;
  status?: string;
  sequence?: string;
}) {
  const { lang, sessionId, runRequest, setRunRequest } = useApp();
  const zh = lang === "zh";
  const [open, setOpen] = useState(false),
    [position, setPosition] = useState<{ x: number; y: number } | null>(null);
  const [size,setSize]=useState<{width:number;height:number}|null>(null);
  const resizing=useRef<{x:number;y:number;width:number;height:number}|null>(null);
  const [messages, setMessages] = useState<Message[]>([]),
    [input, setInput] = useState("");
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [failedMessage, setFailedMessage] = useState("");
  const [actionProgress,setActionProgress]=useState<AssistantProgress[]>([]);
  const [elapsed,setElapsed]=useState(0);
  useEffect(()=>{if(!busy)return;setElapsed(0);const started=Date.now();const timer=window.setInterval(()=>setElapsed(Math.floor((Date.now()-started)/1000)),1000);return()=>clearInterval(timer);},[busy]);
  const panel = useRef<HTMLElement>(null),
    composer = useRef<HTMLTextAreaElement>(null),
    end = useRef<HTMLDivElement>(null);
  const toggle = useRef<HTMLButtonElement>(null);
  const wasOpen = useRef(false);
  const version = useRef(0);
  const drag = useRef<{
    x: number;
    y: number;
    left: number;
    top: number;
  } | null>(null);
  useEffect(() => {
    version.current++;
    setMessages([]);
    setError("");
    setBusy(false);
    setFailedMessage("");
    setActionProgress([]);
  }, [jobId]);
  useLayoutEffect(() => {
    const clamp = () => {
      setSize(s=>s?{width:Math.min(s.width,innerWidth-16),height:Math.min(s.height,innerHeight-88)}:s);
      setPosition((p) =>
        p
          ? {
              x: Math.max(
                8,
                Math.min(innerWidth - panel.current!.offsetWidth - 8, p.x),
              ),
              y: Math.max(
                72,
                Math.min(innerHeight - panel.current!.offsetHeight - 8, p.y),
              ),
            }
          : p,
      );
    };
    clamp();
    window.addEventListener("resize", clamp);
    return () => window.removeEventListener("resize", clamp);
  }, [open]);
  useEffect(() => {
    if (open) composer.current?.focus();
    else if (wasOpen.current) toggle.current?.focus();
    wasOpen.current = open;
  }, [open]);
  useEffect(() => {
    end.current?.scrollIntoView({
      block: "nearest",
      behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth",
    });
  }, [messages, busy, actionProgress]);
  const send = async (value = input, retry = false) => {
    const text = value.trim();
    if (!text || busy) return;
    const token = version.current;
    setInput("");
    if (composer.current) composer.current.style.height = "auto";
    setBusy(true);
    setError("");
    setFailedMessage("");
    setActionProgress([]);
    if (!retry) setMessages((m) => [...m, { role: "user", text }]);
    try {
      const response = await api.chat(text, jobId, sessionId, jobId?undefined:sequence,
        event=>{if(token===version.current)setActionProgress(items=>[...items.filter(item=>!(event.phase==='tool_complete'&&item.phase==='tool_running'&&item.tool===event.tool)),event]);},
        messages.slice(-6).map(item=>({role:item.role==='agent'?'assistant' as const:'user' as const,text:item.text})));
      if (token !== version.current) return;
      setMessages((m) => [
        ...m,
        { role: "agent", text: response.message, codons: response.codon_rows, structure:response.structure, atlas:response.atlas, prepared:response.prepared_sequence, tools:response.tool_calls, fasta:response.fasta_records },
      ]);
      if (response.tool_result && runRequest && jobId) {
        const updated = {
          ...runRequest,
          selection: {
            ...runRequest.selection,
            weights: response.tool_result.weights,
            confidence: response.tool_result.confidence,
          },
        };
        setRunRequest(updated);
        rememberRequest(jobId, updated);
        onChange();
      }
    } catch (e) {
      if (token === version.current) {
        setError(e instanceof Error ? e.message : String(e));
        setFailedMessage(text);
      }
    } finally {
      if (token === version.current) setBusy(false);
    }
  };
  const suggestions =
    phase === 3 && status === "success"
      ? [
          zh ? "解释 Top-K 的选择依据" : "Explain the Top-K selection",
          zh ? "按大肠杆菌优化密码子" : "Optimize codons for E. coli",
        ]
      : phase === 2
        ? [
            zh ? "解释三个评分器的区别" : "Explain the three scorers",
            zh ? "提高活性的配平权重" : "Give activity more weight",
          ]
        : [
            zh
              ? "如何设置突变数量和阶数？"
              : "How do I set the mutation quota?",
            zh
              ? "PheNom 需要什么底物信息？"
              : "What substrate information does PheNom need?",
          ];
  return (
    <aside
      ref={panel}
      className={`copilot ${open ? "expanded" : ""} ${busy ? "is-busy" : ""}`}
      style={{...(position?{left:position.x,top:position.y,right:'auto',bottom:'auto'}:{}),
        ...(open&&size?{width:size.width,height:size.height}:{})}}
      aria-label="Enzora Assistant"
      onKeyDown={(e) => {
        if (e.key === "Escape" && open) {
          e.stopPropagation();
          setOpen(false);
        }
      }}
    >
      <div
        className="copilot-handle"
        tabIndex={0}
        role="toolbar"
        aria-label={
          zh
            ? "拖动助手，或使用方向键移动"
            : "Drag assistant, or move with arrow keys"
        }
        onKeyDown={(e) => {
          if ((e.target as HTMLElement).closest("button")) return;
          const shifts: Record<string, [number, number]> = {
            ArrowLeft: [-20, 0],
            ArrowRight: [20, 0],
            ArrowUp: [0, -20],
            ArrowDown: [0, 20],
          };
          const shift = shifts[e.key];
          if (!shift) return;
          e.preventDefault();
          const box = panel.current!.getBoundingClientRect();
          setPosition({
            x: Math.max(
              8,
              Math.min(innerWidth - box.width - 8, box.left + shift[0]),
            ),
            y: Math.max(
              72,
              Math.min(innerHeight - box.height - 8, box.top + shift[1]),
            ),
          });
        }}
        onPointerDown={(e) => {
          if ((e.target as HTMLElement).closest("button")) return;
          const box = panel.current!.getBoundingClientRect();
          drag.current = {
            x: e.clientX,
            y: e.clientY,
            left: box.left,
            top: box.top,
          };
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const d = drag.current;
          if (!d) return;
          setPosition({
            x: Math.max(
              8,
              Math.min(
                innerWidth - panel.current!.offsetWidth - 8,
                d.left + e.clientX - d.x,
              ),
            ),
            y: Math.max(
              72,
              Math.min(
                innerHeight - panel.current!.offsetHeight - 8,
                d.top + e.clientY - d.y,
              ),
            ),
          });
        }}
        onPointerUp={() => {
          drag.current = null;
        }}
        onPointerCancel={() => {
          drag.current = null;
        }}
      >
        <span className="copilot-emblem" aria-hidden="true">
          <svg viewBox="0 0 28 28" fill="none">
            <path
              d="M5 18 11 8l7 12 5-9M5 10l7 10 6-13 5 12"
              stroke="currentColor"
              strokeWidth="1.3"
            />
            <circle cx="11" cy="8" r="2" fill="currentColor" />
            <circle cx="18" cy="20" r="2" fill="currentColor" />
          </svg>
        </span>
        <div className="copilot-title">
          <strong>Enzora Assistant</strong>
          <span>
            {busy
              ? zh
                ? "正在处理"
                : "Working"
              : isDemo()
                ? zh
                  ? "案例问答 · 只读"
                  : "Case Q&A · read-only"
                : zh
                  ? "设计与你协作"
                  : "Your design companion"}
          </span>
        </div>
        <span className="copilot-mode">MiniMax</span>
        <button
          ref={toggle}
          className="copilot-toggle"
          aria-expanded={open}
          aria-controls="copilot-content"
          aria-label={
            open
              ? zh
                ? "收起助手"
                : "Collapse assistant"
              : zh
                ? "打开助手"
                : "Open assistant"
          }
          onClick={() => setOpen(!open)}
        >
          <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
            <path
              d={open ? "M5 8l5 5 5-5" : "M5 12l5-5 5 5"}
              stroke="currentColor"
              strokeWidth="1.7"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>
      {open && (
        <div id="copilot-content" className="copilot-content">
          <div className="copilot-context">
            <span className="copilot-status-dot" />
            <span>
              {jobId
                ? `${zh ? "当前阶段" : "Current phase"} · ${[zh ? "突变库构建" : "Library construction", zh ? "评分与配平" : "Screening & balance", zh ? "优化与结果" : "Optimization & results"][phase - 1]}`
                : zh
                  ? "开始设计前，也可以先提问"
                  : "Ask a question before starting"}
            </span>
          </div>
          <div className="assistant-quick-tools" role="group" aria-label={zh?'助手工具':'Assistant tools'}>
            {[
              {label:'ESM Atlas',path:'M3 9a6 6 0 1 0 12 0A6 6 0 0 0 3 9m10 4 5 5',prompt:jobId||sequence?(zh?'从 ESM Atlas 查询当前蛋白，展示已存储的结构、蛋白信息和来源链接。':'Query ESM Atlas for the current protein. Show the stored structure, metadata and source link.'):(zh?'我想查询 ESM Atlas，请先让我提供 UniProt 编号、Atlas hash 或蛋白序列。':'I want to query ESM Atlas. Ask me for a UniProt accession, Atlas hash or protein sequence.')},
              {label:zh?'结构':'Structure',path:'M3 5l7-3 7 4-1 9-8 3-6-5zm7-3-2 9m9-5-9 5m8 4-8-4m-6 2 6-2',prompt:zh?'查看当前 WT 输入结构，不运行新预测。':'Inspect the supplied WT structure; do not run a new prediction.'},
              {label:zh?'裁剪区段':'Trim region',path:'M3 4a2 2 0 1 0 4 0 2 2 0 0 0-4 0m0 12a2 2 0 1 0 4 0 2 2 0 0 0-4 0M7 5l11 13M7 15 18 2',prompt:zh?'我想提取成熟蛋白区段，请先告诉我需要指定哪些编号。':'I want to extract a mature protein region. Ask me which inclusive residue interval to use.'},
              {label:zh?'密码子':'Codons',path:'M5 2c0 8 10 8 10 16M15 2c0 8-10 8-10 16M6 4h8M8 8h4M8 12h4M6 16h8',prompt:zh?'为当前蛋白优化密码子；请先询问表达宿主。':'Optimize codons for the current protein. Ask which expression host to use first.'},
            ].map(action=><button type="button" key={action.label} disabled={busy} onClick={()=>void send(action.prompt)} title={action.label}><svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d={action.path} stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg><span>{action.label}</span></button>)}
          </div>
          <div
            className="copilot-messages"
            role="log"
            aria-live="polite"
            aria-relevant="additions text"
          >
            {!messages.length && (
              <div className="copilot-welcome">
                <h3>
                  {zh
                    ? "从建库到表达，逐步确认。"
                    : "Build. Balance. Prepare to express."}
                </h3>
                <p>
                  {zh
                    ? "解释 PheNom、EpHod 与 UniStab，调整评分配比，或为选中候选生成宿主偏好的 CDS。"
                    : "Interpret PheNom, EpHod and UniStab, adjust their balance, or prepare host-specific CDS for selected candidates."}
                </p>
                <div className="copilot-suggestions">
                  {suggestions.map((text) => (
                    <button
                      key={text}
                      disabled={busy}
                      onClick={() => void send(text)}
                    >
                      <span>{text}</span>
                      <span aria-hidden="true">↗</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((message, i) => (
              <div key={i} className={`chat-message ${message.role}`}>
                <span className="message-author">
                  {message.role === "user" ? (zh ? "你" : "You") : "Enzora"}
                </span>
                <p>{message.text}</p>
                {!!message.tools?.length && <div className="assistant-tools-used">{message.tools.map((call,index)=><span key={index}>{call.tool.replace(/_/g,' ')} ✓</span>)}</div>}
                {message.fasta && <button className="copilot-export" onClick={()=>downloadText(message.fasta!.map(r=>`>${r.id}\n${r.sequence}`).join('\n'),'enzora-assistant-sequences.fasta')}>{zh?'导出序列':'Export sequences'} FASTA ↗</button>}
                {message.prepared && <button className="copilot-export" onClick={()=>downloadText(`>WT_region_${message.prepared!.start}-${message.prepared!.end}\n${message.prepared!.sequence}\n`,'enzora-mature-region.fasta')}>{zh?'导出指定成熟区段':'Export specified mature region'} {message.prepared.start}–{message.prepared.end} ↗</button>}
                {message.atlas && <div className="atlas-record-card"><div className="atlas-record-heading"><strong>ESM Atlas</strong><span>Alpha API</span></div>
                  {message.atlas.metadata && <><h4>{String(message.atlas.metadata.protein_name??message.atlas.metadata.accession??message.atlas.metadata.protein_hash)}</h4><dl>{['accession','organism','sequence_length','mean_plddt','ptm'].filter(key=>message.atlas!.metadata![key]!=null).map(key=><div key={key}><dt>{key.replace(/_/g,' ')}</dt><dd>{String(message.atlas!.metadata![key])}</dd></div>)}</dl>
                    {message.atlas.metadata.function!=null&&<p>{String(message.atlas.metadata.function)}</p>}
                    {Array.isArray(message.atlas.metadata.sae_features)&&<details><summary>{zh?'SAE 模型特征':'SAE model features'}</summary>{(message.atlas.metadata.sae_features as {feature_index:number;label:string;description:string;value:number}[]).map(f=><article className="atlas-feature" key={f.feature_index}><strong>{f.label}</strong><span>{f.value.toFixed(3)}</span><p>{f.description}</p></article>)}</details>}
                  </>}
                  {message.atlas.similar_proteins?.map(hit=><button disabled={busy} key={hit.protein_hash} onClick={()=>void send(`Show ESM Atlas protein ${hit.protein_hash}. Retrieve its stored record, not a new prediction.`)}>{hit.protein_name??hit.protein_accession} · {hit.similarity_score.toFixed(3)} ↗</button>)}
                  <div className="atlas-record-actions"><a href={message.atlas.source_url} target="_blank" rel="noopener noreferrer">{zh?'数据来源':'Data source'} ↗</a>{message.atlas.metadata?.sequence&&<button onClick={()=>downloadText(`>${String(message.atlas!.metadata!.accession??message.atlas!.metadata!.protein_hash)}\n${message.atlas!.metadata!.sequence}\n`,'esm-atlas-protein.fasta')}>FASTA ↓</button>}{message.structure?.pdb&&<button onClick={()=>downloadText(message.structure!.pdb!,'esm-atlas-structure.pdb')}>PDB ↓</button>}</div>
                  <p className="atlas-source-note">{zh?'数据库中的预测结构与模型特征，不代表实验验证。':'Stored predicted structures and model features; experimental validation is separate.'}</p>
                </div>}
                {message.structure && <figure className="assistant-structure"><ProteinViewer structure={message.structure}/><figcaption>{message.structure.source} · {zh?'拖动旋转，滚轮缩放':'Drag to rotate · scroll to zoom'}</figcaption></figure>}
                {message.codons && (
                  <button
                    className="copilot-export"
                    onClick={() =>
                      downloadText(
                        serializeCsv(message.codons!, [
                          "mutation",
                          "scientific_name",
                          "optimized_cds_with_stop",
                          "gc_fraction",
                        ]),
                        "enzora-agent-codons.csv",
                        "text/csv;charset=utf-8",
                      )
                    }
                  >
                    {zh ? "导出优化 CDS" : "Export optimized CDS"} ↗
                  </button>
                )}
              </div>
            ))}
            {busy && (
              <div className="assistant-action-progress" role="status"><div><strong>{zh?'正在处理你的请求':'Working on your request'}</strong><span>{elapsed}s</span></div>
                <ol>{(actionProgress.length?actionProgress:[{phase:'planning' as const}]).map((event,index)=>{
                  const names:Record<string,string>={query_esm_atlas:'ESM Atlas',lookup_protein:zh?'检索蛋白':'Protein search',fetch_protein:zh?'获取序列':'Fetch sequence',balance_scores:zh?'应用评分配平':'Apply score balance',get_scorer_evidence:zh?'读取预测证据':'Read scorer evidence',prepare_mature_protein:zh?'提取指定区段':'Extract region',inspect_structure:zh?'载入 WT 结构':'Load WT structure',predict_structure:'ESMFold2',optimize_codons:zh?'优化密码子':'Optimize codons',translate_sequence:zh?'翻译序列':'Translate sequence'};
                  const label=event.tool?names[event.tool]??event.tool:event.phase==='explaining'?(zh?'整理工具结果与下一步建议':'Summarizing results and next steps'):(zh?'结合上下文解析目标':'Interpreting your goal and context');
                  return <li key={index} data-complete={event.phase==='tool_complete'||index<actionProgress.length-1}><span aria-hidden="true">{event.phase==='tool_complete'?'✓':'·'}</span>{label}{event.phase==='tool_running'?(zh?' · 执行中':' · running'):''}</li>;
                })}</ol><p>{zh?'工具完成后会显示可检查和下载的结果。':'Tool outputs will appear here for inspection and download.'}</p>
              </div>
            )}
            {error && (
              <div className="copilot-error" role="alert">
                <strong>
                  {zh
                    ? "暂时无法完成这条消息"
                    : "Could not complete this message"}
                </strong>
                <p>{error}</p>
                <button
                  disabled={busy}
                  onClick={() => void send(failedMessage, true)}
                >
                  {zh ? "重试" : "Retry"}
                </button>
                <Link to="/settings">
                  {zh ? "检查连接设置" : "Connection settings"} ↗
                </Link>
              </div>
            )}
            <div ref={end} />
          </div>
          <form
            className="copilot-compose"
            onSubmit={(e) => {
              e.preventDefault();
              void send();
            }}
          >
            <textarea
              ref={composer}
              aria-label={zh ? "发送给 Agent 的消息" : "Message the agent"}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                e.currentTarget.style.height = "auto";
                e.currentTarget.style.height = `${Math.min(96, e.currentTarget.scrollHeight)}px`;
              }}
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing
                ) {
                  e.preventDefault();
                  void send();
                }
              }}
              placeholder={
                zh
                  ? "询问模型，或描述你的设计目标…"
                  : "Ask about a model, or describe your goal…"
              }
              disabled={busy}
              rows={2}
            />
            <div className="compose-bottom">
              <span>
                {isDemo()
                  ? zh
                    ? "真实 MiniMax · 只读案例，不运行模型"
                    : "Live MiniMax · read-only case, no model runs"
                  : zh
                    ? "Enter 发送 · Shift+Enter 换行"
                    : "Enter to send · Shift+Enter for a new line"}
              </span>
              <button
                className="copilot-send"
                disabled={!input.trim() || busy}
                type="submit"
                aria-label={zh ? "发送消息" : "Send message"}
              >
                <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
                  <path
                    d="M10 15V5m-4 4 4-4 4 4"
                    stroke="currentColor"
                    strokeWidth="1.7"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
            </div>
          </form>
        </div>
      )}
      {open && <button type="button" className="assistant-resize" aria-label={zh?'拖动调整助手大小；方向键也可调整':'Resize assistant by dragging or with arrow keys'}
        onKeyDown={event=>{
          const delta:Record<string,[number,number]>={ArrowLeft:[-24,0],ArrowRight:[24,0],ArrowUp:[0,-24],ArrowDown:[0,24]};
          if(!delta[event.key])return;event.preventDefault();
          const box=panel.current!.getBoundingClientRect();
          setPosition({x:box.left,y:box.top});
          setSize({width:Math.max(300,Math.min(innerWidth-box.left-8,box.width+delta[event.key][0])),height:Math.max(Math.min(300,innerHeight-88),Math.min(innerHeight-box.top-8,box.height+delta[event.key][1]))});
        }}
        onPointerDown={event=>{const box=panel.current!.getBoundingClientRect();setPosition({x:box.left,y:box.top});resizing.current={x:event.clientX,y:event.clientY,width:box.width,height:box.height};event.currentTarget.setPointerCapture(event.pointerId);}}
        onPointerMove={event=>{const r=resizing.current;if(!r)return;const box=panel.current!.getBoundingClientRect();
          setSize({width:Math.max(Math.min(300,innerWidth-16),Math.min(innerWidth-box.left-8,r.width+event.clientX-r.x)),height:Math.max(Math.min(300,innerHeight-88),Math.min(innerHeight-box.top-8,r.height+event.clientY-r.y))});}}
        onPointerUp={()=>{resizing.current=null;}} onPointerCancel={()=>{resizing.current=null;}}><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M6 15l9-9M11 15l4-4" stroke="currentColor" strokeWidth="2"/></svg></button>}
    </aside>
  );
}
