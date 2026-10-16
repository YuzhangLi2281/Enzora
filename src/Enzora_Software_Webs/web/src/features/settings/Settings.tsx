/** Settings: model paths (server-side), Gemini config, API key session,
 *  runtime defaults. The API key lives only in server memory. */

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, API_BASE, isDemo } from '../../shared/api/client';
import { t } from '../../shared/lib/i18n';
import { useApp } from '../../app/store';
import type { ServerSettings } from '../../shared/api/contracts';
import { Badge, ErrorNote, Panel, fmt } from '../../shared/ui/primitives';

export function Settings() {
  const lang = useApp((s) => s.lang);
  const sessionId = useApp((s) => s.sessionId);
  const setSessionId = useApp((s) => s.setSessionId);
  const query = useQuery({ queryKey: ['settings'], queryFn: () => api.getSettings() });
  const keyStatus = useQuery({ queryKey: ['session-status', sessionId], queryFn: () => api.sessionStatus(sessionId), refetchInterval: 30_000 });
  const [draft, setDraft] = useState<ServerSettings | null>(null);
  const [apiKey, setApiKey] = useState('');
  const [replacingKey, setReplacingKey] = useState(false);
  const hasConfiguredKey = keyStatus.data?.configured === true;
  const showConfiguredKey = hasConfiguredKey && !replacingKey;
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (query.data && !draft) setDraft(query.data);
  }, [query.data, draft]);

  const setModel = (model: string, key: string, value: unknown) => {
    setDraft((d) => (d ? { ...d, models: { ...d.models, [model]: { ...d.models[model], [key]: value } } } : d));
  };
  const setLlm = (key: string, value: unknown) => {
    setDraft((d) => (d ? { ...d, llm: { ...d.llm, [key]: value } } : d));
  };

  const save = async () => {
    if (!draft) return;
    setBusy(true);
    setError(null);
    try {
      const saved = await api.putSettings({ models: draft.models, llm: draft.llm, runtime: draft.runtime });
      setDraft(saved);
      setMessage(t('settings.saved', lang));
      setTimeout(() => setMessage(null), 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const storeKey = async () => {
    setBusy(true);
    setError(null);
    try {
      const { session_id } = await api.sessionKey(apiKey.trim());
      setSessionId(session_id);
      setApiKey('');
      setReplacingKey(false);
      setMessage(lang === 'zh' ? '密钥已配置。' : 'Key configured.');
      setTimeout(() => setMessage(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main id="main-content" className="page-shell" style={{ maxWidth: 960 }}>
      <div className="page-heading"><div className="eyebrow">WORKSPACE PREFERENCES</div><h1>{t('settings.title', lang)}</h1><p>{lang === 'zh' ? '前端连接与服务端配置集中查看。' : 'Review your frontend connection and server configuration.'}</p></div>
      <Panel title={lang === 'zh' ? '前端连接' : 'Frontend connection'}>
        <p className="section-note">{lang === 'zh' ? '界面可独立部署，计算任务通过 API 交由服务器执行。' : 'This interface deploys independently; computation runs on the server through its API.'}</p>
        <div className="config-address">{API_BASE}</div>
        <div className="small dim" style={{ marginTop: 8 }}>{lang === 'zh' ? '部署后可修改 config.js 中的 apiBaseUrl，无需重新构建。' : 'Edit apiBaseUrl in config.js after deployment, without rebuilding.'}</div>
        <div style={{ marginTop: 10 }}><Badge kind={isDemo() ? 'warn' : 'accent'}>{isDemo() ? (lang === 'zh' ? '演示模式 · 无真实计算' : 'Demo mode · No live computation') : (lang === 'zh' ? '实时 API 模式' : 'Live API mode')}</Badge></div>
      </Panel>
      <div style={{ height: 16 }} />
      <Panel title={t('settings.title', lang)}>
        {query.isError && <><ErrorNote text={query.error.message} /><button style={{ marginTop: 10 }} onClick={() => void query.refetch()}>{lang === 'zh' ? '重新连接' : 'Reconnect'}</button></>}
        {query.isLoading && <div className="small dim">{t('common.loading', lang)}</div>}
        {draft && (
          <>
            <div style={{ marginBottom: 12 }}>
              <div className="small dim" style={{ marginBottom: 6 }}>model paths (server-side)</div>
              {Object.entries(draft.models).map(([model, cfg]) => (
                <div key={model} style={{ display: 'grid', gridTemplateColumns: '140px 1fr', gap: 8, alignItems: 'center', marginBottom: 6 }}>
                  <span className="badge mono accent">{model}</span>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {['repo', 'python', 'model_dir', 'checkpoint', 'weights_dir', 'esmc_model_dir', 'molt5_model_dir'].filter((k) => cfg && k in (cfg as Record<string, unknown>)).map((k) => (
                      <input key={k} style={{ flex: 1, minWidth: 150 }} value={String((cfg as Record<string, unknown>)[k] ?? '')} onChange={(e) => setModel(model, k, e.target.value)} />
                    ))}

                  </div>
                </div>
              ))}
            </div>

            <div style={{ marginBottom: 12 }}>
              <div className="small dim" style={{ marginBottom: 6 }}>Gemini</div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <input style={{ flex: 2, minWidth: 200 }} value={String(draft.llm.base_url ?? '')} onChange={(e) => setLlm('base_url', e.target.value)} />
                <select aria-label={lang === 'zh' ? '参数模型' : 'Parameter model'} style={{ width: 'auto' }} value={String(draft.llm.parameter_model ?? '')} onChange={(e) => setLlm('parameter_model', e.target.value)}>
                  <option value="gemini-2.5-flash">gemini-2.5-flash</option>
                  <option value="gemini-2.5-pro">gemini-2.5-pro</option>
                </select>
                <select aria-label={lang === 'zh' ? '分析模型' : 'Analysis model'} style={{ width: 'auto' }} value={String(draft.llm.analysis_model ?? '')} onChange={(e) => setLlm('analysis_model', e.target.value)}>
                  <option value="gemini-2.5-flash">gemini-2.5-flash</option>
                  <option value="gemini-2.5-pro">gemini-2.5-pro</option>
                </select>
              </div>
            </div>

            <div className="muted-box" style={{ marginBottom: 12 }}>
              <div className="small dim" style={{ marginBottom: 6 }}>{lang === 'zh' ? 'Gemini API 配置' : 'Gemini API configuration'}</div>
              <p className="small dim">{lang === 'zh' ? '自动读取服务端环境变量或 server/.env 中的 GEMINI_API_KEY，读取成功后自动显示掩码。需要更换时可设置临时会话密钥。' : 'Automatically reads GEMINI_API_KEY from the server environment or server/.env and shows a masked configured value. You can replace it with a temporary session key.'}</p>
              <div style={{ display: 'flex', gap: 8 }}>
                <input aria-label={lang === 'zh' ? 'Gemini 密钥' : 'Gemini API key'} type="password" autoComplete="off" value={showConfiguredKey ? '••••••••••••' : apiKey} readOnly={showConfiguredKey} onChange={(e) => setApiKey(e.target.value)} placeholder="sk-…" />
                {showConfiguredKey ? <button onClick={() => setReplacingKey(true)} disabled={busy}>{lang === 'zh' ? '更换密钥' : 'Replace key'}</button> : <button onClick={() => void storeKey()} disabled={!apiKey.trim() || busy}>{lang === 'zh' ? '设置' : 'Set key'}</button>}
                {replacingKey && <button className="ghost" disabled={busy} onClick={() => { setReplacingKey(false); setApiKey(''); }}>{lang === 'zh' ? '取消' : 'Cancel'}</button>}
              </div>
              <div className="small" style={{ marginTop: 6 }}>
                {keyStatus.isLoading ? <Badge kind="plain">{lang === 'zh' ? '检查配置中…' : 'Checking configuration…'}</Badge>
                  : keyStatus.data?.configured
                  ? <Badge kind="ok">{keyStatus.data.source === 'demo' ? (lang === 'zh' ? '演示密钥状态（模拟）' : 'Demo key status (simulated)') : keyStatus.data.source === 'environment' ? (lang === 'zh' ? '已从服务端环境读取' : 'Configured from server environment') : (lang === 'zh' ? '临时会话密钥已配置' : 'Temporary session key configured')}</Badge>
                  : <Badge kind="warn">{lang === 'zh' ? '尚未配置密钥' : 'No key configured'}</Badge>}
                {keyStatus.isError && <div className="error-text small">{lang === 'zh' ? '无法检查密钥配置，请确认后端连接。' : 'Could not check key configuration. Check the backend connection.'}</div>}
              </div>
            </div>

            <button className="primary" onClick={() => void save()} disabled={busy}>{t('settings.save', lang)}</button>
            {message && <span className="small" style={{ marginLeft: 10, color: 'var(--ok)' }}>{message}</span>}
            {error && <ErrorNote text={error} />}
          </>
        )}
      </Panel>
    </main>
  );
}

export { fmt };
