/** Lazy Plotly view. Resolve CSS tokens before handing values to Plotly. */
import { useEffect, useRef, useState } from 'react';
import { useApp } from '../../app/store';

function resolveTokens(value: unknown, style: CSSStyleDeclaration): unknown {
  if (typeof value === 'string' && value.startsWith('var(--')) return style.getPropertyValue(value.slice(4, -1)).trim();
  if (Array.isArray(value)) return value.map((item) => resolveTokens(item, style));
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, resolveTokens(item, style)]));
  return value;
}

export function PlotWrap(props: {
  data: unknown[];
  layout: Record<string, unknown>;
  className?: string;
  style?: React.CSSProperties;
  onPointClick?: (point: unknown) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const theme = useApp((s) => s.theme);
  const lang = useApp((s) => s.lang);
  const [error, setError] = useState(false);
  const clickRef = useRef(props.onPointClick);
  clickRef.current = props.onPointClick;
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    let disposed = false;
    let loaded: typeof import('plotly.js-dist-min').default | null = null;
    setError(false);
    const style = getComputedStyle(document.documentElement);
    const layout = {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: style.getPropertyValue('--text-dim').trim(), size: 11 },
      ...props.layout,
      margin: { l: 48, r: 16, t: 34, b: 40, ...(props.layout.margin as Record<string, number> | undefined) },
    };
    import('plotly.js-dist-min').then(async ({ default: Plotly }) => {
      if (disposed) return;
      loaded = Plotly;
      await Plotly.newPlot(element, resolveTokens(props.data, style) as unknown[], resolveTokens(layout, style) as Record<string, unknown>, { responsive: true, displaylogo: false });
      if (disposed) return;
      const plotElement = element as HTMLDivElement & { on?: (event: string, callback: (event: unknown) => void) => void };
      plotElement.on?.('plotly_click', (event: unknown) => clickRef.current?.(event));
    }).catch(() => { if (!disposed) setError(true); });
    return () => { disposed = true; loaded?.purge(element); };
  }, [props.data, props.layout, theme]);
  return <div><div ref={ref} className={props.className} style={{ width: '100%', height: 260, ...props.style }} />{error && <div className="error-text small" role="alert">{lang === 'zh' ? '图表加载失败，请刷新页面重试。' : 'The chart could not be loaded. Refresh to try again.'}</div>}</div>;
}

/** Real atom coordinates rendered as a molecular cartoon; no third-party fetch. */
export function ProteinViewer({structure,height=280}:{structure:import('../api/contracts').ProteinStructure;height?:number}) {
  const viewport=useRef<HTMLDivElement>(null);
  const stage=useRef<import('ngl').Stage|null>(null);
  const [error,setError]=useState('');
  const [representation,setRepresentation]=useState<'cartoon'|'ball+stick'>('cartoon');
  const lang=useApp(s=>s.lang),zh=lang==='zh';
  useEffect(()=>{
    if(!structure.pdb||!viewport.current)return;
    let disposed=false;
    let observer:ResizeObserver|undefined;
    setError('');
    void import('ngl').then(async({Stage})=>{
      if(disposed)return;
      const instance=new Stage(viewport.current!,{backgroundColor:'white',tooltip:true});
      stage.current=instance;
      const component=await instance.loadFile(new Blob([structure.pdb!],{type:'text/plain'}),{ext:'pdb'});
      if(disposed){instance.dispose();return;}
      if(!component)throw new Error('Structure could not be parsed');
      component.addRepresentation(representation,{colorScheme:'residueindex',quality:'medium'});
      component.autoView();
      observer=new ResizeObserver(()=>instance.handleResize());observer.observe(viewport.current!);
    }).catch(()=>{if(!disposed)setError(zh?'结构渲染失败，可下载原始 PDB 查看。':'Structure rendering failed. Download the original PDB to inspect it.');});
    return()=>{disposed=true;observer?.disconnect();stage.current?.dispose();stage.current=null;};
  },[structure.pdb,representation]);
  if(!structure.pdb)return <PlotWrap style={{height}} data={[{type:'scatter3d',mode:'lines+markers',x:structure.points.map(p=>p.x),y:structure.points.map(p=>p.y),z:structure.points.map(p=>p.z),text:structure.points.map(p=>`Residue ${p.residue}`),hoverinfo:'text',line:{width:5,color:'#8AC3C1'},marker:{size:2,color:structure.points.map(p=>p.residue),colorscale:[[0,'#8AC3C1'],[1,'#B497C7']]}}]} layout={{margin:{l:0,r:0,t:0,b:0},scene:{aspectmode:'data',xaxis:{visible:false},yaxis:{visible:false},zaxis:{visible:false}},showlegend:false}}/>;
  return <div className="protein-viewer"><div className="protein-viewer-toolbar"><button type="button" aria-pressed={representation==='cartoon'} onClick={()=>setRepresentation('cartoon')}>{zh?'带状':'Cartoon'}</button><button type="button" aria-pressed={representation==='ball+stick'} onClick={()=>setRepresentation('ball+stick')}>{zh?'球棍':'Atoms'}</button><button type="button" onClick={()=>stage.current?.autoView()} aria-label={zh?'重置结构视角':'Reset structure view'}>↺</button></div><div ref={viewport} style={{height,width:'100%',position:'relative'}} aria-label={zh?'可旋转的蛋白结构':'Interactive protein structure'}/>{error&&<p role="alert">{error}</p>}</div>;
}
