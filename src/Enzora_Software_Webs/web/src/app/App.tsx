import { useEffect, useState } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { Header } from './Header';
import { Workbench } from '../features/workbench/Workbench';
import { Wizard } from '../features/experiment/Wizard';
import { History } from '../features/runs/History';
import { Tools } from '../features/tools/Tools';
import { Settings } from '../features/settings/Settings';
import { useApp } from './store';
import { API_BASE, accessToken, isDemo, login } from '../shared/api/client';

export default function App() {
  const [auth, setAuth] = useState<'checking' | 'required' | 'ready' | 'offline'>(isDemo() ? 'ready' : 'checking');
  const [code, setCode] = useState('');
  const [authError, setAuthError] = useState('');
  const [loggingIn, setLoggingIn] = useState(false);
  useEffect(() => {
    if (isDemo()) return;
    let active = true;
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 15_000);
    fetch(`${API_BASE}/auth/status`, { signal: controller.signal }).then(async (response) => {
      if (!response.ok) throw new Error('Connection failed');
      const status = await response.json();
      if (status.required && accessToken()) await login(accessToken());
      if (active) setAuth(status.required && !accessToken() ? 'required' : 'ready');
    }).catch(() => { if (active) setAuth(accessToken() ? 'required' : 'offline'); }).finally(() => window.clearTimeout(timer));
    return () => { active = false; controller.abort(); window.clearTimeout(timer); };
  }, []);
  const theme = useApp((s) => s.theme);
  const lang = useApp((s) => s.lang);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.lang = lang;
  }, [theme, lang]);

  if (auth !== 'ready') return <main className="page-shell" style={{ minHeight: '100vh', display: 'grid', placeItems: 'center' }}>
    <form className="panel" style={{ maxWidth: 460, width: '100%', padding: 32 }} onSubmit={async (event) => {
      event.preventDefault(); setLoggingIn(true); setAuthError('');
      try { await login(code); setAuth('ready'); }
      catch (error) { setAuthError(error instanceof Error ? error.message : String(error)); }
      finally { setLoggingIn(false); }
    }}>
      <div className="eyebrow">ENZORA · RESEARCH WORKSPACE</div>
      <h1>{auth === 'checking' ? '正在连接 / Connecting' : '登录 Enzora'}</h1>
      <p>{auth === 'offline' ? '暂时无法连接模型服务，请稍后重试。' : '输入工作区访问码，开始酶序列设计。'}</p>
      <label>访问码 / Access code<input type="password" autoComplete="current-password" value={code} onChange={(e) => setCode(e.target.value)} required /></label>
      {authError && <p role="alert">{authError}</p>}
      <button type="submit" disabled={loggingIn || auth === 'checking'}>{loggingIn ? '验证中…' : '进入工作区 →'}</button>
      <a href="?demo=1#/workbench" style={{ display: 'block', marginTop: 16 }}>查看演示 / Explore demo</a>
    </form>
  </main>;
  return (
    <>
      <Header />
      <Routes>
        <Route path="/" element={<Navigate to="/workbench" replace />} />
        <Route path="/workbench" element={<Workbench />} />
        <Route path="/new" element={<Wizard />} />
        <Route path="/history" element={<History />} />
        <Route path="/tools" element={<Tools />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/workbench" replace />} />
      </Routes>
    </>
  );
}
