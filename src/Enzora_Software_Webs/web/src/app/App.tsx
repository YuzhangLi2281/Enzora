import { useEffect, useState } from 'react';
import { Link, Navigate, Route, Routes } from 'react-router-dom';
import { Header } from './Header';
import iddlabLogo from '../../../../../resource/models/iddlab-logo.svg';
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
      if (accessToken()) await login(accessToken());
      if (active) setAuth(!accessToken() ? 'required' : 'ready');
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
                  ? '连接酶设计、计算筛选与实验准备。'
                  : 'Connecting enzyme design, computational screening and experimental preparation.'}</p>
              </div>
            </section>
            <section className="teams-about" aria-label={lang==='zh'?'项目介绍':'About the project'}>
              <article><h2>Enzora</h2><p>{lang==='zh'?'把三模型突变库构建、基于预测证据的多目标筛选和宿主编码设计连接为一个可检查、可导出的工作流。':'Connects three-model mutation-library construction, evidence-informed multi-objective screening and host-specific DNA design in one inspectable workflow.'}</p></article>
              <article><h2>Phenom</h2><p>{lang==='zh'?'从蛋白序列和底物分子图构建可解释的酶–底物交互指纹，通过原生 Gaussian head 预测动力学参数及其不确定性。':'Builds interpretable enzyme–substrate interaction fingerprints from protein sequences and molecular graphs. Native Gaussian heads predict kinetic parameters and their uncertainty.'}</p></article>
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
            <section className="teams-acknowledgment" aria-labelledby="teams-ack"><div><h2 id="teams-ack">Acknowledgment</h2></div>
              <div className="acknowledgment-track" tabIndex={0} aria-label={lang==='zh'?'致谢机构，支持横向滚动':'Acknowledged groups; scroll horizontally'}>
                <a href="https://github.com/Intelligent-Drug-Discovery-Lab" target="_blank" rel="noopener noreferrer"><img src={iddlabLogo} alt="Intelligent Drug Discovery Lab" width="92" height="92"/><div><h3>IDDLab</h3><p>{lang==='zh'?'感谢科学指导与项目支持。':'For scientific guidance and project support.'}</p></div><span aria-hidden="true">↗</span></a>
              </div>
            </section>
            <section className="teams-contact" aria-labelledby="teams-contact-title">
              <h2 id="teams-contact-title">Contact</h2>
              <div className="teams-contact-list">
                <a href="mailto:YuzhangLi2281@gmail.com"><strong>Yuzhang Li</strong><span>YuzhangLi2281@gmail.com</span><span aria-hidden="true">↗</span></a>
                <a href="mailto:3108542350@qq.com"><strong>Kexin Li</strong><span>3108542350@qq.com</span><span aria-hidden="true">↗</span></a>
                <a href="mailto:zhaoyupu6@gmail.com"><strong>Yupu Zhao</strong><span>zhaoyupu6@gmail.com</span><span aria-hidden="true">↗</span></a>
                <a href="mailto:ZhaiYuJia-ZHAI@outlook.com"><strong>Yujia Zhai</strong><span>ZhaiYuJia-ZHAI@outlook.com</span><span aria-hidden="true">↗</span></a>
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
