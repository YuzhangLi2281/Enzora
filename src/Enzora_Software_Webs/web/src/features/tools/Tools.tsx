import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { DEMO_ATLAS_QUERY, DEMO_ATLAS_SEQUENCE, demo } from '../../shared/demo/backend';
import { api, isDemo } from '../../shared/api/client';
import { useApp } from '../../app/store';
import { ChatPanel } from '../experiment/ChatPanel';
import { InputFiles } from '../experiment/InputFiles';
import { ProteinViewer } from '../../shared/ui/PlotWrap';
import { downloadText, serializeCsv } from '../../shared/lib/download';
import { Badge, ErrorNote, Loading } from '../../shared/ui/primitives';
import type { AtlasRecord } from '../../shared/api/contracts';

export function Tools() {
  const {lang,sessionId,runRequest}=useApp();
  const zh=lang==='zh';
  const catalog=useQuery({queryKey:['tools'],queryFn:()=>api.tools()});
  const hosts=useQuery({queryKey:['hosts'],queryFn:()=>api.hosts()});
  const runs=useQuery({queryKey:['runs'],queryFn:()=>api.listRuns()});
  const [sequence,setSequence]=useState(isDemo()?DEMO_ATLAS_SEQUENCE:runRequest?.sequence??'');
  const [type,setType]=useState<'protein'|'dna'|'rna'>('protein');
  const [operation,setOperation]=useState('atlas');
  const [atlasQuery,setAtlasQuery]=useState(isDemo()?DEMO_ATLAS_QUERY:'');
  const [similar,setSimilar]=useState(false);
  const [host,setHost]=useState('b_subtilis');
  const [start,setStart]=useState(1),[end,setEnd]=useState(isDemo()?DEMO_ATLAS_SEQUENCE.length:runRequest?.sequence.length??1);
  const [job,setJob]=useState('');
  const [query,setQuery]=useState('');
  const [lookup,setLookup]=useState<Awaited<ReturnType<typeof api.lookup>>['candidates']>([]);
  const [busy,setBusy]=useState(false),[error,setError]=useState('');
  const [result,setResult]=useState<Record<string,unknown>|null>(null);
  const [atlas,setAtlas]=useState<AtlasRecord|null>(null);
  const [structure,setStructure]=useState<Awaited<ReturnType<typeof api.sequenceTool>>['structure']>();
  const [exportRows,setExportRows]=useState<Record<string,unknown>[]>([]);
  const [fasta,setFasta]=useState('');
  useEffect(()=>{
    setResult(null);
    setAtlas(null);
    setStructure(undefined);
    setExportRows([]);
    setFasta('');
    setError('');
    if(isDemo()&&type==='protein'&&operation==='atlas'&&atlasQuery===DEMO_ATLAS_QUERY&&sequence===DEMO_ATLAS_SEQUENCE&&!similar){
      const example=demo.atlas(DEMO_ATLAS_QUERY);setAtlas(example);setStructure(example.structure);setResult(example.metadata!);
      setFasta(`>${String(example.metadata!.accession)}\n${DEMO_ATLAS_SEQUENCE}\n`);
    }
  },[sequence,type,operation,atlasQuery,similar,host,start,end]);
  const perform=async(action:()=>Promise<void>)=>{setBusy(true);setError('');try{await action();}catch(e){setError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}};
  const loadRun=async(id:string)=>{
    setJob(id);
    if(!id)return;
    await perform(async()=>{const resolved=JSON.parse(await api.readArtifact(id,'00_resolved_request.json'));setSequence(resolved.sequence);setEnd(resolved.sequence.length);setType('protein');});
  };
  const run=()=>void perform(async()=>{
    setResult(null);setAtlas(null);setStructure(undefined);setExportRows([]);setFasta('');
    if(operation==='atlas'){const value=await api.atlas(atlasQuery.trim()||sequence,similar);setAtlas(value);setResult(value.metadata??{source:value.source});if(value.structure)setStructure(value.structure);if(value.metadata?.sequence)setFasta(`>${String(value.metadata.accession??value.metadata.protein_hash)}\n${value.metadata.sequence}\n`);return;}
    if(!sequence)throw new Error(zh?'请先输入或导入序列。':'Enter or import a sequence first.');
    if(operation==='translate'){
      if(type==='protein')throw new Error(zh?'翻译工具需要 DNA 或 RNA。':'Translation requires DNA or RNA.');
      const value=await api.translate(sequence,type);setResult(value as unknown as Record<string,unknown>);
      const protein=value.protein_sequence;
      if(protein)setFasta(`>translated_protein\n${protein}\n`);
    }else if(operation==='codon'){
      if(type!=='protein')throw new Error(zh?'密码子优化需要蛋白序列。':'Codon optimization requires a protein sequence.');
      const value=await api.codonOptimize(sequence,[host]);
      if(value.errors.length)throw new Error(value.errors.map(e=>e.error).join('; '));
      setResult({host:value.results[0]?.scientific_name,method:value.method,protein_unchanged:true});
      setExportRows(value.results as unknown as Record<string,unknown>[]);
      setFasta(value.results.map(r=>`>${r.organism_key}_CDS\n${r.optimized_cds_with_stop}`).join('\n'));
    }else{
      if(type!=='protein')throw new Error(zh?'该工具需要蛋白序列。':'This tool requires a protein sequence.');
      const value=await api.sequenceTool(operation==='region'?'prepare_mature_protein':operation,sequence,operation==='region'?{start,end}:{});
      if(value.structure){setStructure(value.structure);setResult({source:value.structure.source,resolved_CA_atoms:value.structure.points.length});}
      if(value.prepared_sequence){const r=value.prepared_sequence;setResult({start:r.start,end:r.end,length:r.sequence.length,source:'user-specified interval',active_run_unchanged:true});setFasta(`>mature_region_${r.start}-${r.end}\n${r.sequence}\n`);}
    }
  });
  const foldReady=catalog.data?.some(t=>t.name==='esmfold2'&&t.health.available);
  const contextRun=runs.data?.find(r=>r.id===job);
  return <>
    <main id="main-content" className="page-shell tools-workspace">
      <header className="tools-heading"><div><h1>Models & Tools</h1><p>{zh?'导入序列，调用工具，检查并导出结果。':'Import a sequence, run a tool, inspect and export its result.'}</p></div><Link to="/workbench">{zh?'打开酶设计工作区':'Open enzyme workspace'} ↗</Link></header>
      {isDemo()&&<div className="tools-notice"><p>{zh?'结构展示案例：UPI000036401E，260个残基，Atlas 已存储预测结构。可重复旋转、缩放和导出，不运行新结构预测。':'Structure example: UPI000036401E · 260 residues · stored Atlas prediction. Rotate, zoom and export repeatedly; no new folding.'}</p><button className="small" disabled={busy} onClick={()=>{setSequence(DEMO_ATLAS_SEQUENCE);setAtlasQuery(DEMO_ATLAS_QUERY);setOperation('atlas');setType('protein');setSimilar(false);setStart(1);setEnd(DEMO_ATLAS_SEQUENCE.length);setJob('');}}>{zh?'载入结构示例':'Load structure example'} ↗</button></div>}
      <div className="tools-operation-grid">
        <section className="tools-input" aria-labelledby="tools-sequence">
          <h2 id="tools-sequence">{zh?'序列输入':'Sequence input'}</h2>
          <div className="tools-input-meta"><label>{zh?'类型':'Type'}<select value={type} disabled={busy} onChange={e=>setType(e.target.value as typeof type)}><option value="protein">Protein</option><option value="dna">DNA</option><option value="rna">RNA</option></select></label><span>{sequence.length} {zh?'字符':'characters'}</span></div>
          <textarea aria-label={zh?'工具输入序列':'Tool input sequence'} value={sequence} rows={6} spellCheck={false} disabled={busy} onChange={e=>{setSequence(e.target.value);setJob('');}} placeholder={zh?'粘贴序列，或从下方导入文件':'Paste a sequence, or import a file below'}/>
          <InputFiles lang={lang} sessionId={sessionId} typeOverride={type} disabled={busy} onBusy={setBusy} onResults={items=>{
            if(items.length!==1||items[0].result.records?.length!==1)throw new Error(zh?'请每次导入一条明确的序列。':'Import one explicit sequence at a time.');
            const r=items[0].result.records![0];setSequence(r.sequence);setType(r.type as typeof type);setEnd(r.sequence.length);setJob('');
          }}/>
          <details className="tools-database"><summary>{zh?'从数据库查找':'Find in a database'}</summary><div><input aria-label={zh?'数据库查询':'Database search'} value={query} onChange={e=>setQuery(e.target.value)} placeholder="UniProt accession / enzyme name"/><button disabled={busy||!query.trim()} onClick={()=>void perform(async()=>{setLookup((await api.lookup(query)).candidates);})}>{zh?'查找':'Find'}</button></div>{lookup.map(c=><button key={c.accession} disabled={busy} onClick={()=>void perform(async()=>{const r=await api.lookupResolve(c.accession,c.source);setSequence(r.sequence);setType('protein');setEnd(r.sequence.length);setJob('');})}>{c.accession} · {c.name} ↗</button>)}</details>
          <div className="tools-action-settings"><h2>{zh?'选择工具':'Choose a tool'}</h2>
            <select aria-label={zh?'工具':'Tool'} value={operation} disabled={busy} onChange={e=>setOperation(e.target.value)}>
              <option value="atlas">ESM Atlas · {zh?"蛋白信息与结构":"Protein information & structure"}</option>
              <option value="codon">{zh?'宿主密码子优化':'Host codon optimization'}</option><option value="translate">{zh?'DNA / RNA 翻译':'DNA / RNA translation'}</option><option value="region">{zh?'提取指定成熟区段':'Extract specified mature region'}</option><option value="inspect_structure">{isDemo()?(zh?'查看已存储示例结构':'Inspect stored example structure'):(zh?'查看输入 WT 结构':'Inspect supplied WT structure')}</option><option value="predict_structure" disabled={!foldReady}>ESMFold2 {foldReady?'':(zh?'· 未配置':'· Not configured')}</option>
            </select>
            {operation==='atlas'&&<><label>{zh?'UniProt ID / Atlas hash（或使用上方序列）':'UniProt ID / Atlas hash (or use the sequence above)'}<input disabled={busy} value={atlasQuery} onChange={e=>setAtlasQuery(e.target.value)} placeholder="P01308"/></label><label className="atlas-similarity-option"><input type="checkbox" disabled={busy} checked={similar} onChange={e=>setSimilar(e.target.checked)}/>{zh?'按序列查询 SAE 特征相似蛋白':'Find SAE feature-similar proteins from a sequence'}</label></>}
            {operation==='codon' &&<label>{zh?'表达宿主':'Expression host'}<select disabled={busy} value={host} onChange={e=>setHost(e.target.value)}>{(hosts.data??[]).map(h=><option key={h.key} value={h.key}>{h.scientific_name}</option>)}</select></label>}
            {operation==='region'&&<><div className="tools-region-fields"><label>{zh?'起始位点（含）':'Start (inclusive)'}<input type="number" disabled={busy} min={1} max={sequence.length} value={start} onChange={e=>setStart(Number(e.target.value))}/></label><label>{zh?'结束位点（含）':'End (inclusive)'}<input type="number" disabled={busy} min={start} max={sequence.length} value={end} onChange={e=>setEnd(Number(e.target.value))}/></label></div><p>{zh?'按你指定的编号提取，不推断信号肽或切割位点；不会替换运行中的序列。':'Uses your residue interval; does not predict signal peptides or cleavage sites, or replace an active run.'}</p></>}
            {operation==='inspect_structure'&&<p>{isDemo()?(zh?'显示与当前序列对应的已保存示例结构，不运行新预测。':'Displays the stored structure matching this example sequence; no new folding.'):(zh?'仅显示与当前序列完全对应的已提供 WT 骨架。预测新结构请使用 ESMFold2。':'Displays the supplied WT backbone only when it matches this sequence. Use ESMFold2 for new predictions.')}</p>}
            <button className="primary" disabled={busy||(!sequence&&!(operation==='atlas'&&atlasQuery.trim()))||(operation==='predict_structure'&&!foldReady)} onClick={run}>{busy?(zh?'正在调用…':'Running…'):(zh?'运行工具':'Run tool')} ↗</button>
          </div>
        </section>
        <section className="tools-result" aria-labelledby="tools-result"><h2 id="tools-result">{zh?'工具结果':'Tool result'}</h2>
          {error&&<ErrorNote text={error}/>} 
          {!result&&!busy&&<div className="tools-result-empty"><strong>{zh?'结果会显示在这里':'Your result appears here'}</strong><p>{zh?'在左侧选好输入和工具，点击“运行工具”。也可以打开 Assistant，通过自然语言调用这些工具。':'Choose your input and tool, then click Run tool. You can also ask Assistant to call these tools.'}</p></div>}
          {busy&&<Loading/>}
          {structure&&<figure><ProteinViewer structure={structure} height={360}/><figcaption>{structure.source} · {zh?'拖动旋转，滚轮缩放':'Drag to rotate · scroll to zoom'}</figcaption>{structure.pdb&&<button onClick={()=>downloadText(structure.pdb!,'enzora-structure.pdb')}>PDB ↓</button>}</figure>}
          {atlas&&<div className="atlas-record-card"><div className="atlas-record-heading"><strong>ESM Atlas</strong><span>Alpha API</span></div><a href={atlas.source_url} target="_blank" rel="noopener noreferrer">{zh?'查看数据来源':'View data source'} ↗</a>
            {!atlas.structure&&atlas.metadata&&<p>{zh?'该条目没有已存储结构，可查看序列和模型特征。':'This record has no stored structure. Sequence and model features are available below.'}</p>}
            {Array.isArray(atlas.metadata?.sae_features)&&<details><summary>{zh?'SAE 特征与残基区域':'SAE features & residue regions'}</summary>{(atlas.metadata.sae_features as {feature_index:number;label:string;description:string;value:number}[]).map(f=><article className="atlas-feature" key={f.feature_index}><strong>{f.label}</strong><span>{f.value.toFixed(3)}</span><p>{f.description}</p></article>)}</details>}
            {atlas.similar_proteins?.map(hit=><button key={hit.protein_hash} onClick={()=>{setAtlasQuery(hit.protein_hash);setSimilar(false);}}>{hit.protein_name??hit.protein_accession} · {hit.similarity_score.toFixed(3)} ↗</button>)}
            {!!atlas.similar_proteins?.length&&<p>{zh?'点击条目后，再运行工具查看详情。相似度为 SAE 特征余弦相似度。':'Choose a hit, then run the tool for its details. Scores are SAE feature cosine similarity.'}</p>}
          </div>}
          {result&&<dl className="tools-result-values">{Object.entries(result).filter(([key,v])=>key!=='sequence'&&typeof v!=='object').map(([key,value])=><div key={key}><dt>{key.replace(/_/g,' ')}</dt><dd>{String(value)}</dd></div>)}</dl>}
          {fasta&&<><pre className="tools-sequence-result">{fasta}</pre><button onClick={()=>downloadText(fasta,'enzora-tool-result.fasta')}>FASTA ↗</button></>}
          {!!exportRows.length&&<button onClick={()=>downloadText(serializeCsv(exportRows,['organism_key','scientific_name','optimized_cds_with_stop','gc_fraction']),'enzora-tool-result.csv','text/csv;charset=utf-8')}>CSV ↗</button>}
          <div className="tools-assistant-context"><h3>Enzora Assistant</h3><label>{zh?'对话关联的运行':'Run context for chat'}<select value={job} disabled={busy} onChange={e=>void loadRun(e.target.value)}><option value="">{zh?'当前输入序列':'Current input sequence'}</option>{(runs.data??[]).filter(r=>r.status==='success'||r.status==='running').map(r=><option key={r.id} value={r.id}>{r.config_summary?.sequence_name??r.id}</option>)}</select></label><p>{zh?'可以询问当前结构，或指定需要提取的残基区间。筛选任务的蛋白与此 Atlas 案例是不同序列。':'Ask about this structure, or specify a residue interval. The screening WT and this Atlas example are different sequences.'}</p></div>
        </section>
      </div>
      <details className="tools-model-catalog"><summary>{zh?'已配置模型与可用状态':'Configured models & availability'}</summary>{catalog.isError&&<ErrorNote text={catalog.error.message}/>}<div>{(catalog.data??[]).map(tool=><article key={tool.name}><strong>{tool.name}</strong><Badge kind={tool.health.available?'ok':'warn'}>{tool.health.available?(zh?'可用':'Ready'):(zh?'未配置':'Not configured')}</Badge><p>{tool.description}</p></article>)}</div></details>
    </main>
    <ChatPanel jobId={job||null} sequence={type==='protein'?sequence:undefined} onChange={()=>void runs.refetch()} status={contextRun?.status} phase={contextRun?.status==='success'?3:contextRun?.progress?.phase??1}/>
  </>;
}
