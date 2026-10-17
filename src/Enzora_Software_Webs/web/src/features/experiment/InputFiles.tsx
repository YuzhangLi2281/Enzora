import { useRef, useState } from 'react';
import { api, isDemo } from '../../shared/api/client';
import type { ClassifyResult, Lang } from '../../shared/api/contracts';

export const INPUT_FILE_ACCEPT = '.fasta,.fa,.faa,.fna,.ffn,.frn,.txt,.seq,.pdb,.ent,.cif,.mmcif,.gb,.gbk,.genbank,.fastq,.fq,.csv,.json,.sbol,.xml,.rdf';

/** Files are parsed by the same server workflow as pasted input. */
export function InputFiles({ lang, sessionId, typeOverride, disabled, onResults, onBusy }: {
  lang: Lang;
  sessionId: string | null;
  typeOverride?: string | null;
  disabled?: boolean;
  onResults: (results: { filename: string; result: ClassifyResult }[]) => void | Promise<void>;
  onBusy?: (busy: boolean) => void;
}) {
  const picker = useRef<HTMLInputElement>(null);
  const [artifact, setArtifact] = useState<ClassifyResult['bioartifact']>();
  const [format, setFormat] = useState('json');
  const [notice, setNotice] = useState('');
  const exportFile = async () => {
    if (!artifact) return;
    setBusy(true); setError(null); setNotice('');
    try {
      const result = await api.exportArtifact(artifact, format);
      const url = URL.createObjectURL(new Blob([result.content], { type: 'text/plain;charset=utf-8' }));
      const link = document.createElement('a');
      link.href = url; link.download = `enzora-input.${format === 'genbank' ? 'gb' : format}`;
      document.body.appendChild(link); link.click(); link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      setNotice(result.warnings.join(' '));
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setBusy(false); }
  };
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const read = async (files: FileList | null) => {
    if (!files?.length) return;
    setBusy(true); onBusy?.(true); setError(null);
    try {
      const results = [];
      for (const file of Array.from(files)) {
        if (file.size > 2 * 1024 * 1024) throw new Error(lang === 'zh' ? `${file.name} 超过 2 MB，请拆分后导入。` : `${file.name} exceeds 2 MB. Split it before importing.`);
        const content = await file.text();
        results.push({ filename: file.name, result: await api.inputFile(file.name, content, sessionId, typeOverride) });
      }
      setArtifact(results.length === 1 ? results[0].result.bioartifact : undefined);
      setNotice('');
      await onResults(results);
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setBusy(false); onBusy?.(false); if (picker.current) picker.current.value = ''; }
  };
  return <div className="input-files">
    <input ref={picker} type="file" multiple accept={INPUT_FILE_ACCEPT} style={{ display: 'none' }} onChange={(e) => void read(e.target.files)} />
    <button type="button" className="ghost small" disabled={disabled || busy} onClick={() => picker.current?.click()}>{busy ? (lang === 'zh' ? '正在读取文件…' : 'Reading files…') : (lang === 'zh' ? '＋ 导入文件' : '＋ Import files')}</button>
    <span className="small dim" style={{ marginLeft: 8 }}>{lang === 'zh' ? 'FASTA · CSV · JSON · GenBank · SBOL · PDB/mmCIF · FASTQ' : 'FASTA · CSV · JSON · GenBank · SBOL · PDB/mmCIF · FASTQ'}</span>
    {artifact && <div style={{ marginTop: 10, display: 'flex', gap: 8, alignItems: 'center' }}>
      <label className="small">{lang === 'zh' ? '导出已导入记录' : 'Export imported records'}
        <select value={format} onChange={e => setFormat(e.target.value)} disabled={busy} style={{ marginLeft: 8 }}>
          {(isDemo()?['json','csv','fasta']:['json', 'csv', 'fasta', 'genbank', 'sbol']).map(f => <option key={f} value={f}>{f.toUpperCase()}</option>)}
        </select>
      </label>
      <button type="button" className="ghost small" disabled={busy} onClick={() => void exportFile()}>{lang === 'zh' ? '导出' : 'Export'}</button>
    </div>}
    {notice && <p className="small" role="status">{notice}</p>}
    {error && <div role="alert" className="error-text small" style={{ marginTop: 6 }}>{error}</div>}
  </div>;
}
