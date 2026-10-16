import { useEffect, useState } from 'react';
import { Link, Navigate, Route, Routes } from 'react-router-dom';
import { Header } from './Header';
import enzoraLogo from '../../../../../resource/models/enzora-logo.svg';
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
        <Route path="/teams" element={
          <main id="main-content" className="page-shell enzora-teams-page">
            <section className="teams-intro">
              <img src={enzoraLogo} alt="" width="88" height="88" />
              <div>
                <p className="teams-project">Tongji-China · Enzora</p>
                <h1>Teams</h1>
                <p>{lang === 'zh'
                  ? 'Enzora 是同济团队酶工程项目的软件工作区，连接突变库设计、多目标筛选与实验准备。这里集中提供项目与团队资源入口。'
                  : 'Enzora is the software workspace of the Tongji-China enzyme-engineering project. It connects mutation-library design, multi-objective screening and experimental preparation.'}</p>
              </div>
            </section>
            <section className="teams-resources" aria-labelledby="teams-repositories">
              <h2 id="teams-repositories">{lang === 'zh' ? '项目仓库' : 'Project repositories'}</h2>
              <div className="teams-resource-list">
                <a href="https://github.com/YuzhangLi2281/Enzora" target="_blank" rel="noopener noreferrer">
                  <div><strong>Enzora</strong><p>{lang === 'zh' ? 'Agent 工作流、模型适配与软件文档。仓库访问可能需要权限。' : 'Agent workflow, model adapters and software documentation. Repository access may require permission.'}</p></div><span aria-hidden="true">↗</span>
                </a>
                <a href="https://github.com/YuzhangLi2281/Phenom.git" target="_blank" rel="noopener noreferrer">
                  <div><strong>Phenom</strong><p>{lang === 'zh' ? 'Phenom 模型仓库入口，访问可能需要权限。' : 'Phenom model repository. Access may require permission.'}</p></div><span aria-hidden="true">↗</span>
                </a>
              </div>
            </section>
            <section className="teams-resources" aria-labelledby="teams-wiki">
              <h2 id="teams-wiki">Tongji-China Wiki</h2>
              <p className="teams-note">{lang === 'zh' ? '预留入口，Software 与 Model 内容定稿发布后启用。' : 'Reserved links · activated after the Software and Model pages are finalized.'}</p>
              <div className="teams-resource-list">
                <a className="teams-pending" href="https://2026.igem.wiki/tongji-china/software" aria-disabled="true" tabIndex={-1} onClick={(event) => event.preventDefault()}>
                  <div><strong>Software</strong><p>{lang === 'zh' ? '软件方法、工作流与使用指南。' : 'Software methods, workflow and user guide.'}</p></div><span className="teams-coming-soon">Coming soon</span>
                </a>
                <a className="teams-pending" href="https://2026.igem.wiki/tongji-china/model" aria-disabled="true" tabIndex={-1} onClick={(event) => event.preventDefault()}>
                  <div><strong>Model</strong><p>{lang === 'zh' ? '模型原理、假设与计算证据。' : 'Model principles, assumptions and computational evidence.'}</p></div><span className="teams-coming-soon">Coming soon</span>
                </a>
              </div>
            </section>
            <nav className="teams-workspace-links" aria-label={lang === 'zh' ? '工作区管理' : 'Workspace utilities'}>
              <Link to="/workbench">{lang === 'zh' ? '返回工作区' : 'Back to Workspace'} <span aria-hidden="true">↗</span></Link>
              <Link to="/history">{lang === 'zh' ? '运行记录' : 'History'}</Link>
              <Link to="/settings">{lang === 'zh' ? '设置' : 'Settings'}</Link>
            </nav>
          </main>
        } />

        <Route path="*" element={<Navigate to="/workbench" replace />} />
      </Routes>
    </>
  );
}
